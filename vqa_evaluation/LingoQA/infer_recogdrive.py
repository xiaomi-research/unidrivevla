import os
import argparse
import pandas as pd
import ray
from PIL import Image
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

from vllm import LLM, SamplingParams

# system message 68.5
# no system message 63.9
DEFAULT_SYSTEM_MESSAGE = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""
DEFAULT_MODEL_PATH = "/path/to/model"
DEFAULT_PARQUET = "val.parquet"
DEFAULT_IMAGE_ROOT = "/path/to/LingoQA/Evaluation/images/val"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--parquet_path", type=str, default=DEFAULT_PARQUET)
    p.add_argument("--image_root", type=str, default=DEFAULT_IMAGE_ROOT)
    p.add_argument("--output_path", type=str, default="./internvl_vllm_preds.csv")
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--preproc_workers", type=int, default=8)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    p.add_argument("--max_model_len", type=int, default=30000)
    p.add_argument("--system_message", type=str, default=DEFAULT_SYSTEM_MESSAGE)
    return p.parse_args()

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def load_images_for_segment(image_root: str, segment_id: str, num_frames: int = 5, max_workers: int = 8) -> List[Image.Image]:
    seg_dir = os.path.join(image_root, segment_id)
    paths = [os.path.join(seg_dir, f"{i}.jpg") for i in range(num_frames)]
    with ThreadPoolExecutor(max_workers=min(len(paths), max_workers)) as exe:
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
        self.sampling_params = SamplingParams(temperature=0.01, top_p=0.001, max_tokens=256)

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

def _preprocess_block(df_block: pd.DataFrame, *, image_root: str, system_message: str, num_frames: int = 5, preproc_workers: int = 8) -> pd.DataFrame:
    rows = []
    for _, row in df_block.iterrows():
        qid = row["question_id"]
        seg = row["segment_id"]
        question = row["question"]
        try:
            images = load_images_for_segment(image_root, seg, num_frames=num_frames, max_workers=preproc_workers)
            image_prefix = "".join(["<image>\n" for _ in images])
            
            prompt = (
                f"<|im_start|>system\n{system_message.strip()}<|im_end|>\n"
                f"<|im_start|>user\n{image_prefix}{question}\nAnswer in a single short sentence.<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        except Exception:
            images = None
            prompt = ""

        rows.append({
            "question_id": qid,
            "segment_id": seg,
            "question": question,
            "prompt": prompt,
            "image_inputs": images
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
            "system_message": args.system_message,
            "num_frames": 5,
            "preproc_workers": args.preproc_workers
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