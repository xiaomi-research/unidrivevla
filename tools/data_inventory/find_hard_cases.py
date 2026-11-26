from nuscenes.nuscenes import NuScenes
import os

# 1. 挂载数据 (静音模式，不打印多余信息)
data_root = os.path.join(os.getcwd(), 'data', 'nuscenes')
nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=False)

print("\n" + "="*50)
print("🕵️ 二作雷达开启：自动盘点 [极度遮挡/高不确定性] 样本")
print("="*50 + "\n")

total_hard_cases = 0

# 2. 我们先遍历 Mini 数据集里的前 5 个场景 (Scene)
for i in range(5):
    scene = nusc.scene[i]
    print(f"🎬 正在扫描场景 {i+1}: {scene['description']}")
    
    # 获取这个场景的第一帧
    current_sample_token = scene['first_sample_token']
    frame_count = 0
    scene_hard_cases = 0
    
    # 3. 顺藤摸瓜：只要还有下一帧，就一直往后遍历 (Sample)
    while current_sample_token != '':
        sample = nusc.get('sample', current_sample_token)
        frame_count += 1
        
        # 4. 遍历这一帧画面里的所有 3D 框 (Annotation)
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            
            category = ann['category_name']
            visibility = ann['visibility_token']
            
            # 🎯 核心逻辑：我们只抓捕“被严重遮挡 (visibility=='1')”的“车或人”
            if ('vehicle' in category or 'human' in category) and visibility == '1':
                scene_hard_cases += 1
                total_hard_cases += 1
                
        # 走向下一帧
        current_sample_token = sample['next']
        
    print(f"   👉 扫过 {frame_count} 帧画面，挖出 {scene_hard_cases} 个高难度目标！")

print("\n" + "="*50)
print(f"🏆 盘点结束！在这 5 个场景中共储备了 {total_hard_cases} 个高不确定性优质样本。")
print("="*50 + "\n")