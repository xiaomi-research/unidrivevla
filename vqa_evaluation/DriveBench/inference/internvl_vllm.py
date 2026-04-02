import os
import warnings
import argparse
import json
import traceback
from typing import Any, Dict

import torch
import ray
import numpy as np
import pandas as pd
from PIL import Image
from packaging.version import Version

from vllm import LLM, SamplingParams

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from utils import replace_system_prompt, load_json_files, save_json, replace_system_prompt_new

assert Version(ray.__version__) >= Version("2.22.0")
warnings.filterwarnings("ignore", message="The `use_flash_attention_2` flag is deprecated")

system_message = """
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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def process_pil_image(image: Image.Image, image_size: int, max_num: int):
    transform = build_transform(input_size=image_size)
    images = dynamic_preprocess(image, image_size=image_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_image(img_path):
    return Image.open(img_path).convert('RGB')

def parse_arguments():
    parser = argparse.ArgumentParser(description='InternVL Multi-GPU Inference with vLLM and Ray')
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
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--max_model_len', type=int, default=30000)
    return parser.parse_args()


class LLMPredictor:
    def __init__(self, model_path: str, system_prompt: str, generation_config: Dict,
                 image_size: int, max_num_tiles: int, corruption: str,
                 gpu_memory_utilization: float, max_model_len: int, data_root: str = '.'):
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len
        )
        
        self.sampling_params = SamplingParams(
            temperature=generation_config["temperature"],
            top_p=generation_config["top_p"],
            max_tokens=generation_config["max_new_tokens"],
            min_tokens=50
        )
        
        self.system_prompt = system_prompt
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.corruption = corruption
        self.data_root = data_root

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        # 使用 Pandas 提取数据，彻底告别 Dict 类型的不稳定
        questions = batch['question'].tolist()
        image_paths_batch = batch['image_path'].tolist()
        batch_size = len(questions)

        vllm_inputs = []
        valid_indices = []

        for i in range(batch_size):
            try:
                sample_question = questions[i]
                raw_img_path = image_paths_batch[i]
                
                # 解码我们在预处理阶段序列化的字符串
                sample_filenames = json.loads(raw_img_path) if isinstance(raw_img_path, str) else raw_img_path

                relative_paths = list(sample_filenames.values())
                image_paths = [
                os.path.join(self.data_root, p)
                    for p in relative_paths if isinstance(p, str)
                ]

                sample_images = []
                for img_path in image_paths:
                    if self.corruption and len(self.corruption) > 1 and self.corruption != 'NoImage':
                        img_path = img_path.replace(
                            'nuscenes/samples', f'corruption/{self.corruption}'
                        )

                    if self.corruption == 'NoImage':
                        blank_np_array = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                        img = Image.fromarray(blank_np_array)
                        sample_images.append(img)
                    else:
                        # 强烈约束机制：图片必须存在，否则立刻终止这个Worker并抛出完整溯源报错
                        if not os.path.exists(img_path):
                            raise FileNotFoundError(f"【严重错误】图像文件不存在: {img_path}")
                        img = load_image(img_path)
                        sample_images.append(img)

                if not sample_images:
                    continue

                updated_system_prompt, image_placeholders = replace_system_prompt_new(
                    self.system_prompt, image_paths
                )
                raw_prompt_text = f"{image_placeholders}{updated_system_prompt}{sample_question}"

                formatted_prompt = (
                    f"<|im_start|>system\n{self.system_prompt.strip()}<|im_end|>\n"
                    f"<|im_start|>user\n{raw_prompt_text}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

                vllm_inputs.append({
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"image": sample_images}
                })
                valid_indices.append(i)

            except Exception as e:
                # 显式捕捉错误并抛出超详细的定位信息
                err_msg = traceback.format_exc()
                raise RuntimeError(
                    f"\n{'='*60}\n"
                    f"🚨 数据处理失败!\n"
                    f"-> 当前问题 (Question): {questions[i]}\n"
                    f"-> 图片字典 (Image Dict): {image_paths_batch[i]}\n"
                    f"-> 详细报错追踪:\n{err_msg}\n"
                    f"{'='*60}"
                )

        if not vllm_inputs:
            batch['pred'] = [""] * batch_size
            return batch

        # 保护推理阶段的异常
        try:
            responses = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params)
        except Exception as e:
            err_msg = traceback.format_exc()
            raise RuntimeError(f"\n🚨 vLLM 生成失败!\n{err_msg}")

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
    }

    print("✅ 正在将 JSON 铺平并转化为 Pandas 结构以规避 PyArrow Schema 崩溃问题...")
    with open(args.data, 'r') as f:
        raw_data = json.load(f)
        
    for item in raw_data:
        # 【核心修复点】将字典暴力转成字符串，PyArrow 就无法去逐个推断并比对内部的 Key 是否一致了
        if 'image_path' in item and isinstance(item['image_path'], dict):
            item['image_path'] = json.dumps(item['image_path'])

    # 转储为 Pandas DataFrame，Pandas 是出了名的对不规则数据包容性好
    df = pd.DataFrame(raw_data)
    ds = ray.data.from_pandas(df)

    # 显式告知 Ray 传递给 Worker 的是 Pandas 格式
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

    print("⏳ 推理完成。开始合并分布式结果...")
    # 【核心修复点】不要使用 ds.write_json()，利用 Pandas 本地写文件即可，1.4k 行对内存毫无负担
    out_df = ds.to_pandas()
    
    # 最后存盘前，再把字符串解析回整洁的字典
    if 'image_path' in out_df.columns:
        out_df['image_path'] = out_df['image_path'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    final_res = out_df.to_dict(orient='records')
    save_json(final_res, args.output)
    
    print(f"🎉 处理完成，无任何异常崩溃！结果已安全保存至 {args.output}")

if __name__ == '__main__':
    main()