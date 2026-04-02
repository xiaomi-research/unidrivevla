import os
import json
import argparse
from typing import List, Dict, Any, Tuple
from itertools import islice

import ray
from PIL import Image

from vllm import LLM, SamplingParams

# ---------------- default paths (you can override via CLI) ----------------
MODEL_PATH = '/path/to/model'
JSON_INPUT_PATH = '/path/to/DriveLM/v1_1_val_nus_q_only.json'
JSON_OUTPUT_PATH = './internvl3_output_new_system.json'
IMAGE_ROOT = '/path/to/nuscenes'
# -------------
system_message = """
You are a smart autonomous driving assistant responsible for analyzing and responding to driving scenarios. You are provided with up to six camera images in the sequence [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT]. Each image has normalized coordinates from [0, 1], with (0,0) at the top left and (1,1) at the bottom right.

Instructions:
1. Answer Requirements:
- For multiple-choice questions, provide the selected answer choice along with an explanation.
- For “is” or “is not” questions, respond with a “Yes” or “No”, along with an explanation.
- For open-ended perception and prediction questions, related objects to which the camera.

2. Key Information for Driving Context:
- When answering, focus on object attributes (e.g., categories, statuses, visual descriptions) and motions (e.g., speed, action, acceleration) relevant to driving safety and decision-making.

Use the images and coordinate information to respond accurately to questions related to perception, prediction, planning, or behavior, based on the question requirements.
"""
def get_image_sizes(image_paths: Dict[str, str], root_dir: str) -> Dict[str, Tuple[int, int]]:
    cam_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
    image_sizes = {}
    for cam in cam_order:
        if cam in image_paths:
            img_path = os.path.join(root_dir, image_paths[cam])
            try:
                with Image.open(img_path) as img:
                    image_sizes[cam] = img.size
            except Exception:
                pass
    return image_sizes

def process_question(question: str) -> str:
    cam_mapping = {
        'CAM_FRONT': 'FRONT VIEW', 'CAM_FRONT_LEFT': 'FRONT LEFT VIEW',
        'CAM_FRONT_RIGHT': 'FRONT RIGHT VIEW', 'CAM_BACK': 'BACK VIEW',
        'CAM_BACK_LEFT': 'BACK LEFT VIEW', 'CAM_BACK_RIGHT': 'BACK RIGHT VIEW'
    }
    prompt_header = "The following images are captured simultaneously from different cameras mounted on the same ego vehicle:\n"
    image_prompts = []
    cam_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
    for cam in cam_order:
        view_name = cam_mapping.get(cam)
        if view_name:
            image_prompts.append(f"<{view_name}>:\n<image>")
    final_prompt = prompt_header + "\n".join(image_prompts) + f"\n{question}"
    return final_prompt

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

@ray.remote
def preprocess_shard(items: List[Dict[str, Any]], image_root: str, system_msg: str) -> List[Dict[str, Any]]:
    out = []
    cam_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
    for it in items:
        uid = it.get("id")
        qtext = it.get("question", "")
        image_paths = it.get("image_paths", {})
        try:
            image_files = []
            missing = False
            for cam in cam_order:
                if cam in image_paths:
                    rel = image_paths[cam]
                    if not rel or not isinstance(rel, str):
                        missing = True
                        break
                    abs_p = os.path.join(image_root, rel)
                    if not os.path.exists(abs_p):
                        missing = True
                        break
                    image_files.append(abs_p)
                else:
                    missing = True
                    break
            if missing or len(image_files) == 0:
                out.append({"id": uid, "raw_question": qtext, "prompt": "", "image_files": None, "skip_reason": "missing_images"})
                continue
            
            q_prompt = process_question(qtext)
            full_prompt = (
                f"<|im_start|>system\n{system_msg.strip()}<|im_end|>\n"
                f"<|im_start|>user\n{q_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            out.append({"id": uid, "raw_question": qtext, "prompt": full_prompt, "image_files": image_files, "skip_reason": None})
        except Exception as e:
            out.append({"id": uid, "raw_question": qtext, "prompt": "", "image_files": None, "skip_reason": f"preproc_error:{e}"})
    return out

@ray.remote(num_gpus=1)
class VLLMWorker:
    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.8, max_model_len: int = 30000,
                 temperature: float = 0.01, top_p: float = 0.001, max_new_tokens: int = 4096, min_new_tokens: int = 50):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len
        )
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            min_tokens=min_new_tokens
        )

    def generate_batch(self, items: List[Dict[str, Any]], gen_batch_size: int) -> List[Dict[str, Any]]:
        results = []
        for sub in chunked(items, gen_batch_size):
            vllm_inputs = []
            ids = []
            questions = []
            for it in sub:
                uid = it["id"]
                prompt_str = it.get("prompt", "")
                raw_q = it.get("raw_question", "")
                img_files = it.get("image_files") or []
                pil_images = []
                try:
                    for p in img_files:
                        pil_images.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    results.append({"id": uid, "question": raw_q, "answer": f"image_load_error:{e}"})
                    continue
                vllm_inputs.append({"prompt": prompt_str, "multi_modal_data": {"image": pil_images}})
                ids.append(uid)
                questions.append(raw_q)
            if not vllm_inputs:
                continue
            try:
                outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
                for i, out in enumerate(outputs):
                    try:
                        txt = out.outputs[0].text
                    except Exception:
                        txt = str(out)
                    results.append({"id": ids[i], "question": questions[i], "answer": txt})
            except Exception as e:
                for uid, q in zip(ids, questions):
                    results.append({"id": uid, "question": q, "answer": f"generate_error:{e}"})
        return results

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--json_input", type=str, default=JSON_INPUT_PATH)
    parser.add_argument("--json_output", type=str, default=JSON_OUTPUT_PATH)
    parser.add_argument("--image_root", type=str, default=IMAGE_ROOT)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--preproc_shards", type=int, default=32)
    parser.add_argument("--gen_batch_size", type=int, default=16)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_model_len", type=int, default=30000)
    args = parser.parse_args()

    if not ray.is_initialized():
        ray.init()

    with open(args.json_input, "r", encoding="utf-8") as fin:
        data = json.load(fin)

    flat_qa_list = []
    for scene_id, scene_data in data.items():
        for frame_id, frame_data in scene_data.get('key_frames', {}).items():
            question_idx = 0
            image_paths = frame_data.get('image_paths', {})
            for section_name, qa_list in frame_data.get('QA', {}).items():
                for qa_pair in qa_list:
                    unique_id = f"{scene_id}_{frame_id}_{question_idx}"
                    flat_qa_list.append({
                        "id": unique_id,
                        "question": qa_pair['Q'],
                        "image_paths": image_paths
                    })
                    question_idx += 1

    total_items = len(flat_qa_list)
    n_shards = max(1, min(args.preproc_shards, total_items))
    shard_size = (total_items + n_shards - 1) // n_shards
    shards = [flat_qa_list[i * shard_size:(i + 1) * shard_size] for i in range(n_shards)]

    futures = [preprocess_shard.remote(s, args.image_root, system_message) for s in shards]
    preproc_lists = ray.get(futures)
    
    preprocessed = []
    for sub in preproc_lists:
        preprocessed.extend(sub)

    workers = [
        VLLMWorker.remote(
            args.model_path,
            args.gpu_memory_utilization,
            args.max_model_len
        )
        for _ in range(args.num_workers)
    ]

    valid_items = [r for r in preprocessed if r.get("skip_reason") is None]
    skipped_items = [r for r in preprocessed if r.get("skip_reason") is not None]

    results = []
    for s in skipped_items:
        results.append({"id": s["id"], "question": s.get("raw_question"), "answer": f"skipped:{s.get('skip_reason')}"})

    buckets = [[] for _ in range(args.num_workers)]
    for idx, item in enumerate(valid_items):
        buckets[idx % args.num_workers].append(item)

    gen_futures = []
    for w, bucket in zip(workers, buckets):
        if bucket:
            gen_futures.append(w.generate_batch.remote(bucket, args.gen_batch_size))

    worker_outputs = ray.get(gen_futures)
    for out in worker_outputs:
        results.extend(out)

    try:
        results.sort(key=lambda x: str(x["id"]))
    except Exception:
        pass

    os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
    with open(args.json_output, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main_cli()