from nuscenes.nuscenes import NuScenes
import os
import json

data_root = os.path.join(os.getcwd(), 'data', 'nuscenes')
nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=False)

print("\n🚀 极速炼金炉启动：提取 O(1) 复杂度的连续不确定性值...")

# 准备给一作交差的数据包
output_data = []
PT_THRESHOLD = 20.0 # 假设 20 个激光点以上我们就认为"非常确定"

for i in range(5): # 依然扫前 5 个场景
    scene = nusc.scene[i]
    current_sample_token = scene['first_sample_token']
    
    while current_sample_token != '':
        sample = nusc.get('sample', current_sample_token)
        
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            category = ann['category_name']
            visibility = ann['visibility_token']
            
            # 只找我们关心的：车/人 + 严重遮挡
            if ('vehicle' in category or 'human' in category) and visibility == '1':
                
                # 🏆 核心魔法：直接白嫖官方算好的 LiDAR 点数！不用自己算！
                lidar_pts = ann['num_lidar_pts']
                
                # 算一个连续的不确定性分数 [0.0 到 1.0]
                # 点越少，不确定性越高；如果点数大于阈值，不确定性降为 0
                if lidar_pts >= PT_THRESHOLD:
                    uncertainty_score = 0.0
                else:
                    uncertainty_score = round(1.0 - (lidar_pts / PT_THRESHOLD), 3)
                
                # 把精炼后的数据打包
                output_data.append({
                    "sample_token": current_sample_token,
                    "annotation_token": ann_token,
                    "category": category,
                    "num_lidar_pts": lidar_pts,
                    "uncertainty_continuous": uncertainty_score
                })
                
        current_sample_token = sample['next']

# 写入文件交差
output_file = os.path.join(os.getcwd(), 'data', 'uncertainty_hard_cases.json')
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=4)

print(f"\n✅ 绝杀！提纯完成！")
print(f"📁 已将 {len(output_data)} 个带有 Float 连续不确定性的样本，光速保存至: {output_file}")
print("你可以直接把这个 JSON 丢给一作了！\n")