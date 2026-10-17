# UniDriveVLA 终版交付与验收说明

交付标识：`UNIDRIVEVLA-20261004-FINAL-CPU`。工作目录：`Code/unidrivevla`。官方基线固定为本仓 `main` commit `a93c175af893b35dc16618e659eca4d18bb1ec86`。

本交付包含实现、使用说明、可执行 CPU 检查及原始证据。验收分为代码/CPU 契约与真实运行两层：CPU 通过不等于完整 GPU 训练、收敛、性能或官方 benchmark 通过。七项原始需求的完整验收仍需下面列出的运行证据；不会因为缺少 GPU 将已发生的 CPU 失败改称“未运行”。

最终源码清单、归档与 wheel 哈希、逐检查退出码、失败历史及版本范围，以 [终版机器可读验收记录](../../.codex-tasks/unidrivevla-refactor/deliverables/final-20261004-v3/ACCEPTANCE.json) 为准。不存在 Git 提交的改造文件由逐文件 SHA256 标识，不冒用 main 的 commit 作为新实现版本。

完整 CPU 套件在 2031 的 `/tmp/uni-integrated-48x6b6ps` 上实际执行，28/28 子检查 exit 0，整体 exit 0；测试源码归档 SHA256 为 `4f961e2b41a9202639c0f0da1523646f37f0ecc52be204a65accc486cd31502e`。v3 交付随后只修正 README 的 Bash 恢复命令占位符，并更新本验收说明的链接和结果文字；全部 Python、配置及运行脚本与该测试快照逐文件一致。两份文档的测试/交付 SHA 差异另列于 `DOCUMENTATION_DELTA.json`，文档、链接和重新打包检查使用 v3 字节。完整 CPU 结果没有被重新标注为 v3 文档的执行结果。

## 1. 原始七项需求对照

| 原始需求 | 现有实现/入口 | 可以验收的证据范围 | 尚未完成的完整验收 |
|---|---|---|---|
| 1. 模型、训练、数据、pipeline、评估解耦；多个/混合数据集训练评估 | `adapters/` policy contract；`train/trainer.py`；`data/sample.py`、`registry.py`、`multidomain.py`；`eval/registry.py`。`MultiProfilePolicy` 共享 Qwen、experts 和词表，保留每个 profile 的 head、processor、标定、horizon；global batch 来源同质 | `multidomain`、`mixture`、`streams`、`supervision`、`eval_schedule`；单/双 rank CPU 调度、来源隔离、输入/时间状态及恢复对照 | 真实多来源 GPU 混训、长序列与收敛；FSDP 的真实多 GPU 行为 |
| 2. 参考 LoongForge、LingBot-VLA，实现性能优化 | 原生 PyTorch trainer；DDP/FSDP2、BF16 参数/通信、FP32 master AdamW、activation checkpointing、可选 compile、device prefetch、异步 DCP、冻结前缀缓存；参考来源见 [ROADMAP](ROADMAP.md) | 优化器数值/恢复、注意力语义、异步失败处理、CPU/Gloo 归约；旧快照的 DAF 单算子测量另列，不能覆盖终版全模型 | 同硬件、数据、精度、effective batch 和目标下的 main/new 端到端吞吐、显存、时延与收敛；没有“全局最优”证据 |
| 3. Bench2Drive、NAVSIM、nuScenes、NeuroNCAP、HUGSIM、PhysicalAI-AV、LEAD 训练评估 | 数据源/离线转换器与 profile；官方 planning/PDMS 入口；CARLA/HTTP/FIFO 客户端，范围详见下一表 | 当前 CPU 接口、标定、坐标与通用开环 metric；真实 mini/info/cache 抽样只归属其原快照 | 全量数据和官方评分；许可数据转换；真实模拟器 rollout。NeuroNCAP/HUGSIM 是评测系统，不伪装成训练 dataset |
| 4. 仓库整体规整 | 根 README 指向新包；配置、依赖、入口与支持矩阵；FlashAttention 可选加载；统一 CPU runner；源码 manifest 和 wheel | 语法、包内运行文件哈希、入口在无源码父 `PYTHONPATH` 下运行、引用审查 | 正式服务器部署未执行；原 main 树保留用于对照。没有未经覆盖地删除 legacy 兼容模块或不同 oracle 约定 |
| 5. 其他特性与后续维护 | 完整冻结权重导出、strict initialize、dtype provenance、单/多 profile exact resume、损坏 checkpoint 拒绝、task validity、运行目录 claim、独立评测目录与失败传播 | checkpoint/恢复/缺字段负例、部分梯度与 RNG/loader 状态、BF16 小模型重载逐位一致、异常后状态恢复 | GPU 长训/设备故障恢复。回归机制减少后续维护风险，不能承诺永久无需优化 |
| 6. VQA 与 planning 混合/迭代训练 | `data/vqa.py`、`tasks.py`；`vqa=combined|alternate`；主 AR tokenizer/collator；同质 rank 调度、连续 VQA cursor、任务权重 | `vqa`、`task_schedule`、`insulation`、`resume` 的 CPU loss/梯度、顺序和恢复；B2D 原生 AR 权重 1.0，nuScenes 0.1 | 真实 VQA 语料质量、GPU 联合前向/反向及生成质量。默认无语料时关闭；`alternate` 是改变更新顺序的显式实验 |
| 7. 训练评估逻辑与 main 一致 | 原生 LiDAR frame/原点、native anchors/归一化、prompt、3740/3670 fixed prefix/PAD、loss 求和、LR micro-step、BF16 参数与通信、原生 nuScenes stage 2 motion 解码；planning metrics 调用相应定义 | 真实 main 源码 CPU oracle：输入/heads、attention output/gradient、motion top-k/quality/ID、metrics、优化器时序、配置与 Gloo；明确是分项契约 | 相同全模型权重与样本下的 CUDA loss、完整梯度、预测及训练收敛；实际 DeepSpeed 版本未锁定。新数据源/混训不是 main 的单数据配方 |

## 2. 数据、benchmark 和接口范围

| 项目 | 实现 | 实际支持范围 | 不得冒充已经通过的项目 |
|---|---|---|---|
| nuScenes | `data/nuscenes.py`、`config/data/nuscenes.yaml`、`profile/nuscenes.yaml`、`eval/planning_metrics.py` | native LiDAR 训练与 planning strict/loose；旧 `nuscenes_ego`；stage 2 `SparseMotionDecoder` 的逐检测 `trajs_3d` / `trajs_score`，`det_coordinate_frame=lidar` | 全模型/全量训练；缺少官方 segmentation PKL 时 loose 不可用；motion 输出存在不等于官方 motion 分数 |
| Bench2Drive | `data/bench2drive.py`、`profile/b2d.yaml`、`closedloop/bench2drive.py`、`scripts/infer/run_b2d.sh` | native 训练与 ST-P3 planning；CARLA Leaderboard agent，main 传感器/JPEG/标定/RoutePlanner/PID、20 Hz 十槽历史 | CARLA rollout/官方得分；Python 3.7 CARLA egg 不能当成 navsim Python 3.9 兼容客户端 |
| NAVSIM 1.1/2.0 | `data/navsim.py`、`adapters/navsim_agent.py`、`eval/navsim.py`、`navsim_replay.py` | sample/cache reader、缺失 task validity 的边界转换；按版本和 split 调用 PDMS/EPDMS 入口 | 未产生终版官方分数；1.1 与 2.0/two-stage 分数不能混报 |
| NeuroNCAP | `serve.py`、`closedloop/neuro_ncap.py` | 六相机 HTTP `/alive`、`/reset`、`/infer`，明确标定/pose/CAN bus/timestamp/command；返回 ego-frame planning | 真实 simulator rollout、官方分数；不是训练数据集 |
| HUGSIM | `closedloop/hugsim.py`、`hugsim_client.py` | 可信本地 `obs_pipe` / `plan_pipe` FIFO；相机/kinematic YAML 核验；轨迹转到 HUGSIM 坐标；失败 marker | 真实 0.25 s simulator tick 与 0.5 s 模型采样 rollout；不是训练数据集 |
| PhysicalAI-AV | `tools/convert_physicalai.py`、`data/physicalai.py`、`config/data/physicalai.yaml` | Python ≥3.11 的独立 SDK 环境离线转换；七相机含 front-tele；navsim 环境只读 cache、planning 训练和 mask-aware ADE/FDE/L2 | 许可的真实数据全量转换；通用 waypoint 指标不叫官方 benchmark；不伪造 det/map/motion/occupancy 标签 |
| LEAD main/Py123D | `tools/convert_lead.py`、`data/lead.py`、`config/data/lead.yaml` | Python 3.10 / Py123D 0.7 的独立转换；split 日志清单；精确当前 IMU 标签原点；cache planning 训练、通用 ADE/FDE/L2 | 未接入 LEAD 自有模型 checkpoint 的官方 CARLA evaluator；不能声称本包得到其官方闭环分数。`main` 与 `cvpr2026` 数据/权重不混用 |

PhysicalAI-AV 和 LEAD 在本包中是有明确 planning 监督范围的数据适配；缺少的任务不产生伪负样本 loss。它们没有在本包中定义正式 stage 2 epoch 配方，默认 `stage2_max_epochs=0` 会拒绝误启动。用户提供真实数据和明确额外监督/配方前，不把该限制隐藏为“支持全部任务”。

外部接口参照 [NVIDIA 数据 devkit](https://github.com/NVlabs/physical_ai_av)、[LEAD main 数据与评测说明](https://github.com/kesai-labs/lead) 和其 [官方 Leaderboard wrapper](https://github.com/kesai-labs/lead/blob/main/src/lead/__main__.py)。本包的通用开环指标与 LEAD 自有模型/CARLA 官方评测是分别定义的接口；没有借用上游自有模型 checkpoint 宣称已评估 UniDriveVLA。

## 3. 版本与证据规则

| 证据标识 | 版本/检查位置 | 原始退出与覆盖 |
|---|---|---|
| `MAIN` | 固定 main `a93c175...` | CPU oracle 从该 Git 对象打包指定源文件；不会从服务器任意变动的 main 目录替换参考 |
| `C28-27` | `raw/closure-integration-20261004-142603`；源码归档 `4b52cdc47400690dd286a6ea5abba1d4c7d685fdbc34d0628810821c723432cf`；2031 `/tmp/uni-integrated-yywmng08` | 28 项入口整体 exit 1：27 通过，`gradient_reduction` 的 mock BF16 绝对差 0.25；完整失败保留 |
| `G-REPAIR` | `raw/closure-integration-20261004-143247`；归档 `a2f89fd7809bae57bdbca7bcc8447d17f95e48b56ab1b34a10200784418c0708`；2031 `/tmp/uni-integrated-485o4ihn` | 仅修正 `tools/check_gradient_reduction.py`；该项真实 exit 0，含双 rank Gloo；与上一版 SHA 差集只有这个 checker |
| `CLI-REPAIR` | `raw/direct-cli-20261004-143744/result.json`，同一 G-REPAIR checker | 梯度检查直接 CLI exit 0；cwd `/tmp`；源码父目录不在 `PYTHONPATH`，仅提供所需 NAVSIM devkit |
| `OLD-REAL` | `raw/closure-native-real-data-20261003-132035`、`...b2d-20261003-132854`、`navsim-supervision-2031-20261003-140204` | 历史 nuScenes mini/B2D info/NAVSIM cache 抽样，不覆盖之后新增的 dtype/归约/motion/报告修改 |
| `GPU-FAILED` | 2027 immutable bundle `setup/codex_gpu_closure_20261003_4623de431adc`；源归档 `0dd63b6ad5d7d9b4f5a7d44a99b9f623789552c9f36642b1a49e64f13ac18339` | 原始 NuScenes stage 1 一个 optimizer step 后，第二 backward 子进程 SIGSEGV/exit -11；不属于当前终版源码的成功 GPU 证据 |
| `FINAL-CPU` | `/tmp/uni-integrated-48x6b6ps`；归档 `4f961e2b41a9202639c0f0da1523646f37f0ecc52be204a65accc486cd31502e`；[ACCEPTANCE.json](../../.codex-tasks/unidrivevla-refactor/deliverables/final-20261004-v3/ACCEPTANCE.json) 中的逐文件 manifest、全部日志 | 同一测试快照 28/28 exit 0，整体 exit 0；v3 仅两份文档变化，其余运行文件哈希完全一致。v3 文档单独通过链接/打包验证，不冒充新的全量 CPU 运行 |

`raw/` 路径均相对于 `Code/.codex-tasks/unidrivevla-refactor/`。所有旧失败、临时检查记录和实验产物保留；final 目录中的证据副本便于交付，但不覆盖这些来源。

## 4. 专项：gradient_reduction 的 0.25 失败

`C28-27` 在 `_mock_future_check()` 中失败，发生在真实 Gloo 检查开始之前。mock 两个用例复用 `peer`；`peer.to(dtype=float32)` 可返回原对象，原地除法将该张量减半，随后污染 BF16 用例。生产 hook 没有为迎合测试而修改；checker 改成非原地 `other_rank.div(2)`。

修复 checker SHA256：`b2171a7abdc012636eb0df5793ba75e973f01f61757edeba3fce58a72e2e56a1`。`G-REPAIR` 与 `CLI-REPAIR` 都是该版本的成功证据，实际覆盖：配置/旧 metadata 回退、异步 future 缩放与 BF16 舍入、真实双 rank Gloo、梯度 dtype/shape/rank 一致性、冻结参数转换和 buffer/Parameter 身份。`FINAL-CPU` 已在单一快照重新运行并通过完整 28 项套件。

该失败已经有修复后的 CPU 通过证据；它既不属于“缺 GPU 未运行”，也不能由 BF16 checkpoint 小模型通过替代。另一方面，这些 Gloo/checkpoint 检查仍不证明完整 Qwen 模型、NCCL、DeepSpeed CUDA 或收敛通过。

## 5. 检查清单和结果分类

CPU runner 的 28 个命名检查全部列出，终版逐项 exit code 从 FINAL-CPU 读取：

| 范围 | 检查名称 | 检查边界 |
|---|---|---|
| 权重/恢复/优化器 | `checkpoint`、`resume`、`master_adamw` | 完整冻结权重、DCP 失败/稀疏 optimizer、dtype map、tied 参数、RNG/partial gradient/loader cursor；小模型/CPU，不是长训 |
| 任务/数据调度 | `vqa`、`task_schedule`、`targets`、`streams`、`mixture`、`multidomain`、`supervision` | 输入、loss/梯度、同质来源、未知监督、时序状态与异常恢复；不代表实际 VQA 质量 |
| main 分项 oracle | `image_pipeline`、`official_heads`、`planning_metrics`、`insulation`、`attention_contract`、`native_sources`、`native_policy`、`sequences`、`motion_decoder` | 配置、token/PAD、坐标、指标与合成数值；主模型完整 loss/梯度/预测没有由此验收 |
| 评测/入口 | `closedloop`、`external_sources`、`external_eval`、`eval_schedule`、`b2d_closedloop`、`run_paths`、`eval_failures`、`optional_attention` | HTTP/FIFO/路径/配置/CLI 和合成 cache；不启动模拟器或产生官方分数 |
| 分布式归约 | `gradient_reduction` | CPU/Gloo 数学、类型、配置、未来对象与参数转换；不是 NCCL/DeepSpeed runtime 的等价实验 |

已失败且保留：旧 Hydra defaults int 错误、首个 GPU bundle 的 AF_UNIX 路径过长（0 个已观察更新）、正确短 TMPDIR bundle 的真实 SIGSEGV -11、上表 CPU checker 失败、两次本机等待器 missing-worker/exit 125。失败是否修复及所属源版本单独列于机器记录；不删除后再报“从未失败”。

未运行或未验收：当前终版完整模型的 GPU stage 1/2、冻结前缀与全部参数梯度集合、真实混合数据/VQA 更新、完整 main 数值/收敛、同预算性能、NAVSIM 正式 PDMS/EPDMS、CARLA/NeuroNCAP/HUGSIM 官方 rollout、许可 PhysicalAI/LEAD 全量转换及正式部署。

## 6. 变更说明

- 修复冻结 VLM/LoRA 丢失、缺失权重静默接受、DCP 冷 optimizer 状态、任务/VQA cursor 和 partial-gradient 恢复；严格标记完整 checkpoint。
- 恢复 native LiDAR 原点、anchors/归一化、main fixed-prefix PAD 可见性、B2D AR/aggregate loss、native LR micro-step 与实际梯度裁剪配置；legacy ego 预设独立保留。
- nuScenes native stage 2 补齐无参数 main motion decoder。DDP 的通信精度与全部浮点参数转换显式配置；FSDP 路径保持其单独定义。
- 新导出保存计算参数 dtype；推理先验证完整映射再转换 `.data`，不改变共享参数身份或 buffer dtype；旧无映射文件保持原加载行为。
- 验证与 prediction 独立清空/恢复时序状态和各模块模式，包括异常；来源切换、未标注任务和图像几何均有边界检查。
- 新 run 在 Hydra 写入前 claim 输出；新评测保存独立目录、传播真实失败；旧实验与 checkpoint 保留。
- FlashAttention 只在显式构造其模块时加载，基础 requirements 不自动构建它；新检查器可从正常文档路径直接启动。
- PhysicalAI/LEAD SDK 转换与 navsim 训练环境分离；新格式/接口说明见 [README](../README.md) 和 [canonical contract](../data/sample.py)。

## 7. 使用与复现

现有训练/推理继续使用 `/data/yongkangli/miniconda3/envs/navsim`。本交付未安装依赖、未重装 mmcv、未混用 navsim20、未修改两台正式代码目录。以下训练命令用于准备好的新副本/后续授权运行，本次终版交付不会启动它们。

```bash
conda activate navsim
cd /path/to/delivered/Code/unidrivevla
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# CPU 检查，不初始化训练任务；输出目录必须不存在
python -B tools/run_checks.py --output /tmp/unidrivevla-final-cpu-new-id
python -B tools/check_gradient_reduction.py

# Stage 1：显式使用新的实验名；补齐本数据集的 infos/data/anchors/occ/VAE 环境变量
GPUS=4 DATA=nuscenes bash scripts/train/run_unidrivevla_train.sh experiment_name=ns_s1_new_id
GPUS=4 DATA=b2d bash scripts/train/run_unidrivevla_train.sh experiment_name=b2d_s1_new_id
NAVSIM_TREE=1.1 GPUS=4 DATA=navsim bash scripts/train/run_unidrivevla_train.sh experiment_name=nav11_s1_new_id
NAVSIM_TREE=2.0 GPUS=4 DATA=navsim bash scripts/train/run_unidrivevla_train.sh experiment_name=nav20_s1_new_id

# Stage 2：路径来自实际 Stage 1 输出；使用另一个新实验目录
STAGE1_CHECKPOINT=/path/to/exp/ns_s1_new_id/last.ckpt \
  GPUS=4 DATA=nuscenes bash scripts/train/run_unidrivevla_train.sh stage=stage2 experiment_name=ns_s2_new_id

# 精确恢复：明确旧 run；仅原来按 exact_resume 保存的完整状态可以恢复
GPUS=4 DATA=nuscenes bash scripts/train/run_unidrivevla_train.sh \
  experiment_name=ns_s1_new_id train.resume=true train.exact_resume=true dataloader.params.num_workers=0

# 异构来源使用多 profile agent；来源、权重与依赖先按配置准备
NAVSIM_TREE=1.1 GPUS=4 DATA=multidomain bash scripts/train/run_unidrivevla_train.sh experiment_name=mixed_new_id

# combined 复用 main AR hook；alternate 是明确改变训练顺序的实验
python train.py data=nuscenes vqa=combined 'vqa.jsonl_paths=[/path/to/qa.jsonl]' experiment_name=qa_combined_new_id
python train.py data=nuscenes vqa=alternate 'vqa.jsonl_paths=[/path/to/qa.jsonl]' experiment_name=qa_alternate_new_id
```

`global_batch=128` 与 micro-batch=4 时，4 卡累积 8 次；改变到 batch=1 的 smoke 会改变批内 prompt/时序语义，不作为 main 配方等价证据。NAVSIM stage 2 没有官方 epoch 配方，需要显式实验覆盖 `+stage2_max_epochs`；它不是 main recipe。SDK 转换、开环推理、CARLA/HTTP/FIFO 和旧 checkpoint 的 frame/motion 命令在 README 各专节给出。

## 8. 后续 GPU 与真实 benchmark 验收

1. 首先观察已经登记的等待/诊断状态，不创建第二个执行者。2027 必须四卡都无计算进程、每卡 free ≥23000 MiB 且利用率 <5%；启动前再确认。
2. 已授权的一次追加 GDB 仍针对原失败的 immutable bundle：核对 controller 哈希、旧进程身份以及 lock/receipt/log/preload 缺失，再以独占新日志启动固定 controller，最多 3 个 optimizer step。任何标记存在就只观察，不能重复启动。
3. 保留 native traceback、inferior 实际 signal/exit code、每 rank receipt。GDB 自身 exit 0 不等于模型通过；目前 allocator 进入错误报告路径时 libgcc 回溯加载二次崩溃，原始 CUDA 错误/OOM 根因尚未确认。
4. 排除阻塞后，为终版源码另建独立物理 bundle，先 CPU preflight/配置解析，再做 stage 1/2、冻结前缀、nuScenes/B2D/NAVSIM 两版本、mixed/VQA 短冒烟；逐任务记录真实退出码与源码/数据/checkpoint 哈希。
5. 相同权重、真实样本、完整官方 anchors、batch=4 与一致有效 batch 下，对照 main 的 loss、全部梯度、预测和重载结果；固定实际 DeepSpeed 与 CUDA/NCCL 版本后才验收第 7 项。
6. 同预算测量 warmup 后吞吐、峰值显存、step/prefill/planning latency 与收敛；成功后再跑官方分版本 PDMS/EPDMS、CARLA/NeuroNCAP/HUGSIM rollout，使用真实许可数据验证外部 source。此交付不扩张“一次 GDB”授权为完整训练授权。

代码终版与 CPU 契约验收可交付；全量效果/性能/官方分数验收仍有明确开放项。任务表保留这些开放项，不为生成“终版”而将七项父任务全部改成 DONE。
