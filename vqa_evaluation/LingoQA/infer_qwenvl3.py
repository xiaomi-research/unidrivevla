import os
import argparse
import pandas as pd
import ray
from PIL import Image
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from vllm import LLM, SamplingParams

SYSTEM_PROMPT = """
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
# 2b
# system 65.1
# no system 65.5
# 8b
# system 69.4
# no system 70.0
MODEL_PATH = "/path/to/UniDriveVLA_nuScenes_hf"
IMAGE_ROOT = "/path/to/LingoQA/Evaluation/images/val"
LINGOQA_TEST_PATH = "val.parquet"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--parquet_path", default=LINGOQA_TEST_PATH)
    parser.add_argument("--image_root", default=IMAGE_ROOT)
    parser.add_argument("--output_path", default="qwenvl_vllm_preds_8b_final_0331.csv")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--preproc_workers", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--max_model_len", type=int, default=30000)
    return parser.parse_args()

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def load_images_for_segment(image_root: str, segment_id: str, num_frames: int = 5) -> List[Image.Image]:
    seg_dir = os.path.join(image_root, segment_id)
    paths = [os.path.join(seg_dir, f"{i}.jpg") for i in range(num_frames)]
    with ThreadPoolExecutor(max_workers=min(len(paths), 8)) as exe:
        imgs = list(exe.map(load_image, paths))
    return imgs

class VLLMPredictor:
    def __init__(self, model_path: str, gpu_memory_utilization: float, max_model_len: int):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len
        )
        self.sampling_params = SamplingParams(
            temperature=0.01,
            top_p=0.001,
            max_tokens=256
        )

    def __call__(self, batch: Dict[str, list]) -> Dict[str, list]:
        inputs_all = []
        valid_indices = []
        
        for i in range(len(batch["prompt"])):
            prompt = batch["prompt"][i]
            images = batch["image_inputs"][i]
            if prompt and images is not None:
                inputs_all.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": images}
                })
                valid_indices.append(i)
        
        answers = [""] * len(batch["prompt"])
        
        if inputs_all:
            outputs = self.llm.generate(inputs_all, sampling_params=self.sampling_params)
            for idx, out in zip(valid_indices, outputs):
                try:
                    text = out.outputs[0].text
                except Exception:
                    try:
                        text = out.text
                    except Exception:
                        text = str(out)
                answers[idx] = text
                
        batch["answer"] = answers
        return batch

def _preprocess_block(df_block: pd.DataFrame, *, image_root: str, model_path: str) -> pd.DataFrame:
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    
    rows = []
    for _, row in df_block.iterrows():
        qid = row["question_id"]
        seg = row["segment_id"]
        question = row["question"]

        try:
            images = load_images_for_segment(image_root, seg, num_frames=5)
            messages = [
                # {
                #     "role": "system",
                #     "content": SYSTEM_PROMPT
                # },
                {
                    "role": "user",
                    "content": (
                        [{"type": "image", "image": img} for img in images]
                        + [{"type": "text", "text": question}]
                    ),
                }
            ]

            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            imgs_processed, _ = process_vision_info(
                messages, 
                image_patch_size=getattr(processor, "image_processor").patch_size
            )
        except Exception:
            prompt = ""
            imgs_processed = None

        rows.append({
            "question_id": qid,
            "segment_id": seg,
            "question": question,
            "prompt": prompt,
            "image_inputs": imgs_processed,
        })

    return pd.DataFrame(rows)

def main():
    args = parse_args()

    if not ray.is_initialized():
        ray.init()

    df = pd.read_parquet(args.parquet_path)
    df = df[["question_id", "segment_id", "question"]]

    ds = ray.data.from_pandas(df).repartition(args.num_gpus * 4)

    ds_pre = ds.map_batches(
        _preprocess_block,
        fn_kwargs={
            "image_root": args.image_root,
            "model_path": args.model_path
        },
        batch_format="pandas",
        batch_size=args.batch_size
    )

    ds_out = ds_pre.map_batches(
        VLLMPredictor,
        fn_constructor_kwargs={
            "model_path": args.model_path,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len
        },
        batch_format="pandas",
        batch_size=args.batch_size,
        concurrency=args.num_gpus,
        num_gpus=1
    )

    final_df = ds_out.to_pandas()
    final_df = final_df[["question_id", "segment_id", "question", "answer"]]
    final_df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    main()