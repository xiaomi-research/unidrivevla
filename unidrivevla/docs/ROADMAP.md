# UniDriveVLA 改造与验收

工作根目录是 `Code/`，新训练包是 `Code/unidrivevla/`。官方对照固定为本仓 `main` 的 `a93c175`。外部仓库的能力只作参考，只有接入本包并通过相应检查才计入支持范围。

逐需求验收、版本/证据清单及完整限制见 [ACCEPTANCE.md](ACCEPTANCE.md)；机器记录保留每项真实退出码。

## 当前支持与下一步

| 项目 | Code 中的实现 | 尚需验收 |
|---|---|---|
| nuScenes | main native LiDAR 坐标/原点、fixed prefix PAD 语义、训练 profile、strict/loose 开环 metric helper；真实 mini train/val 样本对照 | 完整 trainval 的真实 loss/梯度/预测和 GPU 对照；loose 依赖官方 planning segmentation PKL |
| Bench2Drive | main native LiDAR 坐标/原点、fixed prefix PAD 语义、数据与 ST-P3 开环 metric helper；真实 info 样本对照；CARLA Leaderboard agent 和十槽历史缓存 | 全量数据和真实 GPU 对照；CARLA rollout 与官方分数 |
| NAVSIM 1.1 / 2.0 | sample cache、profile、agent、官方 PDMS / EPDMS 子进程入口 | 分版本运行真实缓存和正式评分；不能把 1.1、2.0 或不同 split 的分数混报 |
| MultiProfilePolicy | 一个共享 Qwen/VLM 与 perception/action experts；每个 profile 私有 processor 与 planning head；每个 global batch 采一个 profile | 真实多来源 GPU 混训及最终分数；不能用单 profile 的 CPU fixture 代替收敛验证 |
| VQA + planning | main JSONL/collator、combined / alternate 调度、VQA-only forward；MultiProfile 共用语言骨干 | 真实 VQA 语料、多卡训练和生成质量；alternate 改变训练配方 |
| PhysicalAI-AV / LEAD | PhysicalAI profile 覆盖官方七相机；SDK 专用环境转换成 canonical cache；LEAD 标签使用当前精确 IMU pose；训练环境只读 cache；提供 masked waypoint ADE/FDE/L2 | 使用许可的真实数据完成全量转换、训练和官方评估；通用开环误差不是官方分数 |
| NeuroNCAP / HUGSIM | NeuroNCAP HTTP Model Node、HUGSIM local FIFO client；提供 nuScenes 六相机标定和坐标转换 | 真实 simulator rollout 与官方评分；目前只有合成请求/坐标/协议 CPU 回归 |
| checkpoint / exact resume | 新模型导出包含冻结权重、resolved profile/frame 和架构；单与多 profile 的 `train.exact_resume` 保存每 rank RNG、采样/task、temporal 和 partial-gradient 状态；CPU 双 rank中断恢复对照通过 | GPU 长训/硬件故障恢复仍需验证；旧 checkpoint 要显式指定缺失的架构/frame 元数据 |
| main parity / 性能 | nuScenes/B2D 输入、head prompt、main fixed prefix caps/PAD attention、loss 与指标有逐字段或合成数值 oracle | 完整训练/预测 parity、收敛、吞吐、峰值显存未与 main 同预算对照；没有“全局最优”的证据 |

NeuroNCAP 与 HUGSIM 是闭环评测系统；PhysicalAI-AV 与 LEAD 提供训练数据。它们不应全部被实现成同一种 dataset loader。新 converter 只写确实存在的监督并标记有效任务；没有标注的数据不会伪造检测、地图、运动或 occupancy 标签。合成协议回归说明转换公式与消息边界符合预期，不等于真实数据覆盖或 benchmark 通过。

## 七项工作

1. **解耦**：保留 dataset canonical sample → adapter transform → policy → trainer → evaluator 的分工。VQA 使用专用 collator；不同任务采用各 rank 一致的调度。混合来源由 `MultiProfilePolicy` 共享 Qwen/experts、隔离 profile heads，并按同质 global batch 选来源；样本不能在一个 batch 内异构。
2. **性能**：同一硬件/精度/数据/目标/effective batch 下，记录数据等待、VLM、专家、optimizer、checkpoint 时间和峰值显存。先测原配置，再分别评估 compile、通信、I/O 与冻结前缀缓存；每次优化都复查 loss、梯度和最终分数。
3. **数据与 benchmark**：按表中状态推进真实适配，保留官方协议与版本；不以 stub 或合成示例宣称完整支持。
4. **仓库规整**：配置、依赖、运行命令、检查脚本与支持矩阵同步；代码改动只落 Code。凭据、模型、实验输出不入库。
5. **维护机制**：完整 checkpoint 和损坏文件拒绝加载；可定位的失败日志、源码 hash、随机/采样/恢复状态、持续 oracle 回归。单 / 多 profile 都有 opt-in exact resume，要求训练配置、rank/world/device 一致且 planning/VQA worker 数为 0；默认 resume 是 best-effort。
6. **VQA/planning**：默认 `vqa=disabled`；提供 VQA 数据后显式选择 `vqa=combined`，按 main 的 AR hook 在同一次规划前向携带独立 AR batch；`vqa=alternate` 为显式实验。纯 VQA 仅优化加权 AR loss，冻结 VLM 时禁止启用。
7. **main 一致性**：官方输入、坐标、prompt、loss 求和、optimizer 实际执行路径与指标逐项对照；改进选项显式开启。完整模型数值和收敛尚未验证时，不宣称完全等价。

## 已加入的正确性措施

- stage 2 导出和新 DCP 保存已学习的冻结 VLM/LoRA；旧 DCP 通过格式标记区分，新格式缺权重直接失败。
- 新 `last.ckpt` / `best.ckpt` 附带 resolved `agent_config`，用于推理端选择模型类、profile、LoRA/专家结构和 stage-2 motion architecture；旧导出没有该元数据时调用方必须指定兼容配置。
- GT padding 是下限，超出原 128 框/64 地图线后扩容；批内按 mask 还原每个样本。
- nuScenes/B2D 默认保留 main 的命令文本读取；B2D 保留 main aggregate loss 的求和行为，另保留消融开关。
- 默认 micro-batch 4；clipping=1 来自 main 实际使用的 DeepSpeed JSON，而非被绕过的 MMCV hook 配置。
- 混合来源的 visit ID 会在重采样和 epoch 切换时改变；当前为正确性采取整 batch 缓存重置，有吞吐证据后再做逐槽 masking。
- `MultiProfilePolicy` 只注册一份共享的 Qwen/VLM/experts 参数；profile heads 独立，sampling/eval 时显式选择来源。单 profile checkpoint 迁移要求 `agent.initial_profile`，其余目标 profile heads 保留新初始化值。
- PhysicalAI-AV / LEAD cache writer 校验 sample、split 和资产路径；external evaluator 只按 mask 汇总通用 waypoint 指标。
- NeuroNCAP images 使用 `torch.load(weights_only=True)` 和尺寸上限；HUGSIM 只接受实际 FIFO inode，pickle 仅用于本地可信 simulator IPC；出错会写 failure marker，不能当成有效模型 rollout。
- `tools/run_checks.py` 按独立进程运行 CPU 回归并逐项记录 exit code、日志和 `results.json`；它不是 GPU 或真实 benchmark 验收。
- nuScenes / Bench2Drive 默认 profile 使用 main 的 raw LiDAR frame 和固定 prefix caps（3740 / 3670），并保留可见 PAD attention 行为；`*_ego` 是旧 ego-axis checkpoint 的兼容预设。Anchor 目录按 frame 隔离；子集 / 零 motion 类 anchors 只能用于 smoke construction。
- Native nuScenes / Bench2Drive profile 的 LR scheduler 与 main 一样按 micro-iteration 前进；warmup micro-step horizon 根据实际 `grad_accum` 换算，保持同一 global batch 下的 warmup sample budget。Legacy ego / NAVSIM / multidomain profiles 保持 optimizer-step interval。
- train/infer 在 Hydra 写配置前为每次执行 claim 新 output path；覆盖 Hydra output location 的参数会拒绝。手工多 rank 必须显式共享本次 fresh `UNIDRIVE_RUN_ID` / `UNIDRIVE_INVOCATION_ID`。子进程评估错误会使主命令以非零退出并保留错误日志。
- 闭环服务为无 metadata 的旧 checkpoint 提供显式 `checkpoint_frame`；ego 来源选择完整 legacy 预设，stage-2 另声明 motion 架构。新 checkpoint 的 frame/架构 metadata 不能被冲突的 CLI 或环境变量覆盖。
- 额外验证和开环预测隔离时序 runtime 与各模块 train/eval 状态，异常也恢复原状态；不改 main 帧间 bank 算法。完整模型的 GPU 恢复快照开销仍待测量。
- 原生 nuScenes stage 2 补齐 main 的无参数 motion 解码与逐检测输出；规划输出坐标不变，检测/motion 输出显式标注 head 原生 frame。
- 原生 nuScenes/B2D 的 DDP 配方明确选择 BF16 参数和通信，冻结浮点参数也转换而 buffer 保留原 dtype；其它预设保持旧默认。可配置 FP32/native 通信，FSDP2 仍使用其原 FP32 归约。
- FlashAttention 只在显式选择其模块时加载，基础依赖安装不触发可选 CUDA 扩展构建。
- 训练导出保存计算参数 dtype，严格推理恢复完整映射并保留共享参数与 buffer dtype；旧 checkpoint 保持原加载行为。FSDP 的执行精度与 FP32 master 存储精度分别记录。

## 参考项目

- [LoongForge-Embodied](https://github.com/baidu-baige/LoongForge/tree/master/loongforge/embodied) 与其[训练状态/优化说明](https://loongforge.readthedocs.io/en/latest/embodied_tutorial/overview.html)：参考模块化训练、状态完整性和受控性能对照，不照搬其他模型的加速倍数。
- [LingBot-VLA](https://github.com/Robbyant/lingbot-vla)：机器人任务的 Qwen2.5-VL/action expert 系统；数据归一化和模块划分可参考，其工具链版本不直接替换当前 Qwen3-VL/PyTorch 2.6 环境。
- DDP 的 BF16 配方参照本仓固定 main 的 `zero_configs/adam_zero1_bf16.json`；参数与梯度 dtype 语义另核对 [DeepSpeed engine](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/engine.py) 和 [ZeRO-1/2 implementation](https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py)。这些链接是官方当前源码，main 未锁定 DeepSpeed runtime 版本；CPU 检查只验明确的参数转换/归约契约，完整数值一致性仍需实际运行版本的对照。
- [NeuroNCAP Model Node API](https://github.com/atonderski/neuro-ncap/blob/main/docs/framework.md) 与 [UniAD reference server](https://raw.githubusercontent.com/wljungbergh/UniAD/main/inference/server.py)：相机内外参、CAN bus、timestamp、command 和 output frame。
- [HUGSIM closed_loop.py](https://github.com/hyzhou404/HUGSIM/blob/main/closed_loop.py)、[HUGSIM camera config](https://github.com/hyzhou404/HUGSIM/blob/main/configs/sim/nuscenes_camera.yaml) 与 [sim_utils.py](https://github.com/hyzhou404/HUGSIM/blob/main/sim/utils/sim_utils.py)：FIFO、相机几何和 LiDAR waypoint 约定。
- [PhysicalAI-AV toolkit](https://github.com/NVlabs/physical_ai_av)、[PhysicalAI-AV data](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)、[LEAD](https://github.com/kesai-labs/lead)：离线数据转换来源；训练端不装这些 SDK。

执行状态、失败与原始验证日志保存在 `Code/.codex-tasks/unidrivevla-refactor/`。新增数据集和性能任务持续记录为待完成；没有“永久不再优化”或“已达到全局最优”的无证据承诺。
