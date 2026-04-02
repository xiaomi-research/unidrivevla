import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import warnings
import argparse
import json
import traceback
from typing import Any, Dict, List

import torch
import ray
import numpy as np
import pandas as pd
from PIL import Image
from packaging.version import Version

from vllm import LLM, SamplingParams

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from utils import replace_system_prompt, load_json_files, save_json, replace_system_prompt_new

assert Version(ray.__version__) >= Version("2.22.0")
warnings.filterwarnings("ignore", message="The `use_flash_attention_2` flag is deprecated")

system_message = """
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

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=getattr(processor, "image_processor", None) and getattr(processor.image_processor, "patch_size", None),
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs or {}
    }


def load_image(img_path: str) -> Image.Image:
    return Image.open(img_path).convert('RGB')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--system_prompt', type=str, required=True)
    parser.add_argument('--num_processes', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=448)
    parser.add_argument('--max_num_tiles', type=int, default=12)
    parser.add_argument('--corruption', type=str, default='')
    parser.add_argument('--data_root', type=str, default='.', help='Root directory of DriveBench dataset')
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top_p', type=float, default=0.001)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--repetition_penalty', type=float, default=1.05)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--max_model_len', type=int, default=30000)
    return parser.parse_args()


class LLMPredictor:
    def __init__(self, model_path: str, system_prompt: str, generation_config: Dict,
                 image_size: int, max_num_tiles: int, corruption: str,
                 gpu_memory_utilization: float, max_model_len: int, data_root: str = '.'):
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load AutoProcessor from {model_path}: {e}")

        tp_size = 1
        try:
            tp_size = max(1, torch.cuda.device_count())
        except Exception:
            tp_size = 1

        try:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=tp_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=max_model_len,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to construct vLLM LLM for {model_path}: {e}")

        self.sampling_params = SamplingParams(
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            max_tokens=generation_config["max_new_tokens"],
            repetition_penalty=generation_config["repetition_penalty"],
        )

        self.system_prompt = system_prompt
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.corruption = corruption
        self.data_root = data_root

        self.view_map = {
            "CAM_FRONT": "<FRONT_VIEW>",
            "CAM_FRONT_LEFT": "<FRONT_LEFT_VIEW>",
            "CAM_FRONT_RIGHT": "<FRONT_RIGHT_VIEW>",
            "CAM_BACK_LEFT": "<BACK_LEFT_VIEW>",
            "CAM_BACK_RIGHT": "<BACK_RIGHT_VIEW>",
            "CAM_BACK": "<BACK_VIEW>"
        }

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        questions = batch['question'].tolist()
        image_paths_batch = batch['image_path'].tolist()
        batch_size = len(questions)

        vllm_inputs = []
        valid_indices = []

        max_pixels_limit = self.image_size * self.image_size * self.max_num_tiles

        for i in range(batch_size):
            try:
                sample_question = questions[i]
                raw_img_path = image_paths_batch[i]

                sample_filenames = json.loads(raw_img_path) if isinstance(raw_img_path, str) else raw_img_path
                
                sample_images_data = []
                image_paths_for_prompt = []

                for cam_key, rel_path in sample_filenames.items():
                    if not isinstance(rel_path, str):
                        continue
                    
                    img_path = os.path.join(self.data_root, rel_path)
                    image_paths_for_prompt.append(img_path)

                    if self.corruption and len(self.corruption) > 1 and self.corruption != 'NoImage':
                        img_path = img_path.replace('nuscenes/samples', f'corruption/{self.corruption}')

                    if self.corruption == 'NoImage':
                        blank_np_array = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                        img = Image.fromarray(blank_np_array)
                        sample_images_data.append({"img": img, "view": self.view_map.get(cam_key, "")})
                    else:
                        if not os.path.exists(img_path):
                            raise FileNotFoundError(f"图像文件不存在: {img_path}")
                        img = load_image(img_path)
                        sample_images_data.append({"img": img, "view": self.view_map.get(cam_key, "")})

                if not sample_images_data:
                    continue

                updated_system_prompt, _ = replace_system_prompt_new(
                    self.system_prompt, image_paths_for_prompt
                )
                
                system_msg = {"role": "system", "content": updated_system_prompt.strip()}
                user_content = []
                for item in sample_images_data:
                    if item["view"]:
                        user_content.append({"type": "text", "text": item["view"]})
                    user_content.append({
                        "type": "image", 
                        "image": item["img"],
                    })
                
                user_content.append({"type": "text", "text": sample_question})
                user_msg = {"role": "user", "content": user_content}
                messages = [system_msg, user_msg]

                prepared = prepare_inputs_for_vllm(messages, self.processor)

                vllm_input = {
                    "prompt": prepared["prompt"],
                    "multi_modal_data": prepared["multi_modal_data"],
                    "mm_processor_kwargs": prepared.get("mm_processor_kwargs", {})
                }

                vllm_inputs.append(vllm_input)
                valid_indices.append(i)

            except Exception as e:
                err_msg = traceback.format_exc()
                raise RuntimeError(
                    f"\n数据处理失败\n"
                    f"当前问题: {questions[i]}\n"
                    f"详细报错追踪:\n{err_msg}\n"
                )

        if not vllm_inputs:
            batch['pred'] = [""] * batch_size
            return batch

        try:
            responses = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
        except Exception as e:
            err_msg = traceback.format_exc()
            raise RuntimeError(f"\nvLLM 生成失败\n{err_msg}")

        full_predictions = [""] * batch_size
        for idx, resp in zip(valid_indices, responses):
            try:
                full_predictions[idx] = resp.outputs[0].text
            except Exception:
                try:
                    full_predictions[idx] = resp.text
                except Exception:
                    full_predictions[idx] = str(resp)

        batch['pred'] = full_predictions
        return batch


def main():
    args = parse_arguments()

    if not ray.is_initialized():
        ray.init()

    with open(args.system_prompt, 'r') as f:
        system_prompt = f.read()

    generation_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
    }

    with open(args.data, 'r') as f:
        raw_data = json.load(f)

    for item in raw_data:
        if 'image_path' in item and isinstance(item['image_path'], dict):
            item['image_path'] = json.dumps(item['image_path'])

    df = pd.DataFrame(raw_data)
    ds = ray.data.from_pandas(df)

    ds = ds.map_batches(
        LLMPredictor,
        fn_constructor_args=(
            args.model,
            system_prompt,
            generation_config,
            args.image_size,
            args.max_num_tiles,
            args.corruption,
            args.gpu_memory_utilization,
            args.max_model_len,
            args.data_root
        ),
        batch_format="pandas",
        num_gpus=1,
        concurrency=args.num_processes,
        batch_size=16,
    )

    out_df = ds.to_pandas()

    if 'image_path' in out_df.columns:
        out_df['image_path'] = out_df['image_path'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    final_res = out_df.to_dict(orient='records')
    save_json(final_res, args.output)

if __name__ == '__main__':
    main()