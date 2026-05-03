import os
import json

print("\n🧬 启动数据缝合舱：正在将 Uncertainty 与 DriveLM 语义进行基因重组...")

# 1. 配置文件路径
data_dir = os.path.join(os.getcwd(), 'data')
hard_cases_path = os.path.join(data_dir, 'uncertainty_hard_cases.json')
drivelm_path = os.path.join(data_dir, 'v1_1_train_nus.json') # 已经换成你刚下的 Train 真实文件

# 2. 加载我们的 2644 个高遮挡样本
with open(hard_cases_path, 'r') as f:
    hard_cases = json.load(f)
print(f"📦 成功加载本地难例样本：{len(hard_cases)} 个")

# 3. 加载 DriveLM 标注数据
with open(drivelm_path, 'r') as f:
    drivelm_data = json.load(f)
print("📚 成功加载 DriveLM 语义全卷！")

# 4. 构建极其高效的 O(1) 搜索字典
print("\n🔍 正在构建 DriveLM 极速检索索引...")
drivelm_lookup = {}

for scene_token, scene_data in drivelm_data.items():
    if 'key_frames' in scene_data:
        for frame_token, frame_data in scene_data['key_frames'].items():
            # 🌟 核心破局点：官方的键名改成了大写的 'QA'！
            if 'QA' in frame_data:
                drivelm_lookup[frame_token] = frame_data['QA']

print(f"✅ 索引构建完成！共提取到 {len(drivelm_lookup)} 帧的语义问答。")

# 5. 执行缝合手术
match_count = 0
for item in hard_cases:
    token = item['sample_token']
    
    # 如果这个高遮挡样本刚好在 DriveLM 里有标注
    if token in drivelm_lookup:
        item['drivelm_semantic_qa'] = drivelm_lookup[token]
        match_count += 1
    else:
        item['drivelm_semantic_qa'] = []

# 6. 保存最终的神级数据集
output_path = os.path.join(data_dir, 'unidrive_vla_multimodal_hard_cases.json')
with open(output_path, 'w') as f:
    json.dump(hard_cases, f, indent=4)

print(f"\n🎉 基因重组完成！")
print(f"🎯 成功为 {match_count} 个高遮挡难例注入了真实的 DriveLM 语义灵魂！")
print(f"💾 终极多模态数据集已保存至: {output_path}")