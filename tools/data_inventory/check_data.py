from nuscenes.nuscenes import NuScenes
import os

# 这里的逻辑是：只要你在 UniDriveVLA 根目录下运行代码，它就能精准找到数据
current_dir = os.getcwd()
data_root = os.path.join(current_dir, 'data', 'nuscenes')

print("\n" + "="*50)
print(f"🚀 二作专属数据探针启动")
print(f"📂 正在从规范工程路径加载数据: {data_root}")
print("="*50 + "\n")

try:
    # 加载数据集
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)

    # 抽取第一个场景验证
    my_scene = nusc.scene[0]
    print(f"\n✅ 成功读取场景信息: {my_scene['description']}")

    # 提取第一帧并触发可视化
    first_sample_token = my_scene['first_sample_token']
    print("\n👁️ 正在渲染第一帧的雷达点云与3D框，如果顺利，屏幕上会弹出一张图片！")
    nusc.render_sample(first_sample_token)

except Exception as e:
    print(f"\n❌ 数据加载失败，请检查数据是否已经正确拖拽到了 {data_root} 目录下！")
    print(f"详细报错信息: {e}")