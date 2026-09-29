import torch
import torch.nn as nn
import numpy as np
import os
import glob
from safetensors.torch import load_file
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS

# NOTE: Legacy detector kept in repo; this project uses detectors/unidrivevla.py for SparseDrive pipeline.
from mmdet.models.detectors.base import BaseDetector
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLConfig
from qwen_vl_utils import process_vision_info


NUSCENES_SYSTEM_PROMPT = """
Generalist Autonomous Driving Agent
Role: You are an advanced, multimodal AI brain for an autonomous vehicle, capable of Perception, Reasoning, and Planning. Your goal is to drive safely, follow instructions, and deeply understand the dynamic world around you.

Context & Coordinate System
- Ego-Centric View: You are at the origin (0,0). The X-axis represents the lateral distance (perpendicular), and the Y-axis represents the longitudinal distance (forward).
- Inputs: You receive multi-view visual observations (<FRONT_VIEW>, <BACK_VIEW>, etc.), historical ego-motion, and vehicle states (velocity, acceleration).

Core Capabilities
1. **Driving & Planning**:
   - Objective: Generate a safe, comfortable, and feasible 3-second trajectory (6 waypoints, 0.5s interval).
   - Constraints: Strictly adhere to traffic rules, avoid collisions, and respect kinematic limits.
   - Output Format: A sequence of coordinates [(x1,y1), ..., (x6,y6)].

2. **Reasoning & VQA** (Chain-of-Thought):
   - Tasks: Analyze traffic scenes, explain causal logic (e.g., "Why stop?"), identify hazards, and answer queries about the environment (weather, road layout, traffic lights).
   - Reasoning: Break down complex scenarios into step-by-step logic, grounding your answers in visual evidence.

3. **Instruction Following & Grounding**:
   - Tasks: Execute navigation commands (e.g., "Park behind the red truck") and ground textual descriptions to specific visual regions or objects.

4. **Perception & World Modeling** (Future & Current State):
   - Tasks: Detect and track objects, predict their future motion, and estimate 3D occupancy or scene geometry (Gaussian Splatting/Occ).
   - Understanding: Map semantic elements (lanes, crossings) and dynamic agents into a coherent world model.

Instructions
- For **Planning** tasks: Output the "Trajectory".
- For **QA/Reasoning** tasks: Provide a clear, logical, and helpful text response.
- For **Perception** tasks: Output structured descriptions or specific formats as requested.

Always prioritize safety and clarity in your responses.
"""

NUSCENES_USER_PROMPT_TEMPLATE = """
As an autonomous driving system, predict the ego vehicle's future trajectory based on:
1. Surround-view camera images.
2. Historical ego motion over the last {T} steps in the 2D BEV frame:
   {hist_traj_str}
3. Ego status (Velocity and Acceleration):
   {ego_status_str}
4. Active navigation command: [{nav_cmd}].

Output requirements:
- Predict 6 future waypoints over 3.0 seconds.
- Each waypoint format: (x:float, y:float), in meters, relative to (0.00, 0.00).
- Use [PT, ...] to encapsulate the trajectory, e.g.
  [PT, (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)].
- Keep numeric values to 2 decimal places.
""".strip()

@DETECTORS.register_module()
class UniDriveVLM(BaseDetector):
    def __init__(
        self,
        model_path,
        task_loss_weight=dict(planning=1.0),
        planning_head=None,
        **kwargs,
    ):
        super().__init__()
        self.task_loss_weight = task_loss_weight
        
        config = Qwen2_5_VLConfig.from_pretrained(model_path)
        self.vlm_model = Qwen2_5_VLForConditionalGeneration(config)
        
        print(f"Loading weights manually from safetensors...")
        import glob
        safetensor_files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        
        state_dict = {}
        if len(safetensor_files) == 0:
            bin_files = sorted(glob.glob(os.path.join(model_path, "*.bin")))
            if len(bin_files) > 0:
                print(f"Found .bin files, loading using torch.load...")
                for file in bin_files:
                    sd = torch.load(file, map_location="cpu")
                    for k, v in sd.items():
                        if "action_preprocessor.normalizer" in k:
                            continue
                        
                        if k.startswith("visual."):
                            new_key = f"model.{k}"
                        elif k.startswith("model."):
                            new_key = k.replace("model.", "model.language_model.", 1)
                        else:
                            new_key = k
                        state_dict[new_key] = v
            else:
                raise FileNotFoundError(f"No .safetensors or .bin weights found in {model_path}")
        else:
            for file in safetensor_files:
                sd = load_file(file, device="cpu")
                for k, v in sd.items():
                    if "action_preprocessor.normalizer" in k:
                        continue
                
                    if k.startswith("visual."):
                        new_key = f"model.{k}"
                    elif k.startswith("model."):
                        new_key = k.replace("model.", "model.language_model.", 1)
                    else:
                        new_key = k
                    
                    state_dict[new_key] = v

        missing, unexpected = self.vlm_model.load_state_dict(state_dict, strict=False)
        print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(missing) > 0:
            print(f"Missing keys example: {missing[:20]}")
        if len(unexpected) > 0:
            print(f"Unexpected keys example: {unexpected[:20]}")
            
        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=518400)
        
        self.system_prompt = NUSCENES_SYSTEM_PROMPT
        self.user_prompt_template = NUSCENES_USER_PROMPT_TEMPLATE

        self.view_tokens = [
            "<FRONT_VIEW>", "<FRONT_LEFT_VIEW>", "<FRONT_RIGHT_VIEW>", 
            "<BACK_LEFT_VIEW>", "<BACK_RIGHT_VIEW>", "<BACK_VIEW>"
        ]
        
        self.sensor_order = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK'
        ]
        
        self.command_mapping = {0: "TURN RIGHT", 1: "TURN LEFT", 2: "GO STRAIGHT"}

        print("Moving model to CUDA...")
        self.vlm_model = self.vlm_model.to(dtype=torch.bfloat16, device='cuda')

    def extract_feat(self, img):
        return None

    def simple_test(self, img, img_metas, **kwargs):

        pred = self.vlm_model(
            img=img,
            img_metas=img_metas,
            **kwargs,
        )

        result = [dict()]
        result[0]["planning"] = pred
        if isinstance(img_metas, list) and len(img_metas) > 0 and isinstance(img_metas[-1], dict):
            if "sample_idx" in img_metas[-1]:
                result[0]["token"] = img_metas[-1]["sample_idx"]
        return result

    def aug_test(self, imgs, img_metas, **kwargs):
        img = imgs[0]
        metas = img_metas[0]
        return self.simple_test(img, metas, **kwargs)

    def forward(self, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)

    def forward_train(self, **kwargs):
        raise NotImplementedError("UniDriveVLM is inference-only. Please do not call forward_train.")

    @torch.no_grad()
    def forward_test(
        self,
        img=None,
        img_metas=None,
        command=None,
        vel=None,
        acc=None,
        hist_traj=None,
        traj=None,
        sdc_planning=None,
        sdc_planning_mask=None,
        **kwargs,
    ):
        device = next(self.vlm_model.parameters()).device
        
        batch_size = len(img_metas)
        results = []

        for i in range(batch_size):
            meta = img_metas[i]
            
            current_meta = None
            if isinstance(meta, list):
                current_meta = meta[-1]
            elif isinstance(meta, dict):
                queue_keys = [k for k in meta.keys() if isinstance(k, int)]
                if queue_keys:
                    current_meta = meta[sorted(queue_keys)[-1]]
                else:
                    current_meta = meta
            else:
                raise TypeError(f"Unsupported meta type: {type(meta)}")

            curr_hist_traj = hist_traj[i] if hist_traj is not None else None

            if torch.is_tensor(curr_hist_traj):
                curr_hist_traj = curr_hist_traj.cpu().numpy()
                
            curr_hist_traj = np.squeeze(curr_hist_traj) 
            if curr_hist_traj.ndim > 2:
                curr_hist_traj = curr_hist_traj.reshape(-1, 2)
            if curr_hist_traj.ndim == 1:
                curr_hist_traj = curr_hist_traj.reshape(-1, 2)
                
            hist_pos = np.cumsum(curr_hist_traj, axis=0)
            hist_pos = hist_pos - hist_pos[-1:]
                
            hist_str = f"[PT_HIST, {', '.join([f'({float(p[0]):.2f}, {float(p[1]):.2f})' for p in hist_pos])}]"


            v = vel[i]
            if torch.is_tensor(v): v = v.cpu().numpy()
            v = v.flatten()
            vx, vy = float(v[0]), float(v[1])


            a = acc[i]
            if torch.is_tensor(a): a = a.cpu().numpy()
            a = a.flatten()
            ax, ay = float(a[0]), float(a[1])

                
            ego_status = f"- Velocity: ({vx:.2f}, {vy:.2f}) m/s\n   - Acceleration: ({ax:.2f}, {ay:.2f}) m/s^2"
            
            curr_cmd = command[i].item() if command is not None else 2
            cmd_str = self.command_mapping.get(curr_cmd, "GO STRAIGHT")


            image_paths = []
            if 'filename' in current_meta and isinstance(current_meta['filename'], list):
                image_paths = current_meta['filename']
            elif 'cams' in current_meta:
                for sensor in self.sensor_order:
                    if sensor in current_meta['cams']:
                        image_paths.append(current_meta['cams'][sensor]['data_path'])
            
            if not image_paths:
                raise ValueError(f"Cannot find image paths in meta. Keys: {current_meta.keys()}")

            
            user_content_list = []
            
            for view_token, img_path in zip(self.view_tokens, image_paths):
                user_content_list.append({"type": "text", "text": f"{view_token}\n"})
                user_content_list.append({"type": "image", "image": img_path})
                user_content_list.append({"type": "text", "text": "\n"})
            
            final_user_prompt = self.user_prompt_template.format(
                T=len(hist_pos) if curr_hist_traj is not None else 0,
                hist_traj_str=hist_str,
                ego_status_str=ego_status,
                nav_cmd=cmd_str
            )
            user_content_list.append({"type": "text", "text": final_user_prompt})

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content_list}
            ]

            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            inputs = inputs.to(device)

            generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=512)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            traj_pred_np = self.parse_xy(output_text)
            traj_pred_tensor = torch.tensor(traj_pred_np).to(device=device, dtype=torch.float32)

            res_dict = dict(
                planning_gt=dict(
                    sdc_planning=sdc_planning[i] if sdc_planning is not None else None,
                    sdc_planning_mask=sdc_planning_mask[i] if sdc_planning_mask is not None else None,
                    command=command[i] if command is not None else None,
                ),
                result_planning=dict(
                    sdc_traj=traj_pred_tensor.unsqueeze(0)
                ),
            )
            results.append(res_dict)

        final_res = []
        for i, res in enumerate(results):
            entry = dict(planning=res)
            
            meta = img_metas[i]
            if isinstance(meta, list):
                curr = meta[-1]
            elif isinstance(meta, dict):
                q_keys = [k for k in meta.keys() if isinstance(k, int)]
                if q_keys:
                    curr = meta[sorted(q_keys)[-1]]
                else:
                    curr = meta
            
            if "sample_idx" in curr:
                entry["token"] = curr["sample_idx"]
                
            final_res.append(entry)
            
        return final_res

    def parse_xy(self, text, num_points=6):
        try:
            import re
            if "[PT," not in text:
                matches = re.findall(r'\[PT, (.*?)\]', text)
                if not matches:
                    return np.zeros((num_points, 2))
                content = matches[-1]
            else:
                start_idx = text.rfind("[PT,")
                end_idx = text.find("]", start_idx)
                content = text[start_idx + 4 : end_idx]

            points = []
            parts = content.split(')')
            for part in parts:
                part = part.replace('(', '').replace(',', ' ').strip()
                if not part: continue
                vals = [float(x) for x in part.split() if x]
                if len(vals) >= 2:
                    points.append(vals[:2])
            
            arr = np.array(points)
            if len(arr) < num_points:
                pad = np.zeros((num_points - len(arr), 2))
                if len(arr) > 0:
                    pad[:] = arr[-1]
                arr = np.vstack([arr, pad])
            return arr[:num_points]
            
        except Exception:
            return np.zeros((num_points, 2))