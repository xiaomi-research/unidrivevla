# LEAD 闭环评测

使用 [LEAD 固定版本 3764f79](https://github.com/kesai-labs/lead/tree/3764f79698922896638638df8062f2e064034610) 的 CARLA 环境、路线和评分入口。该环境运行传感器、定位和官方 `WaypointTracker`。UniDriveVLA 在训练环境内启动本地推理服务。

服务要求当前 LEAD 训练导出的完整 `last.ckpt`，包括 `agent_config`、六相机配置以及 8×0.25 秒的 ego-frame 轨迹规格。前端用 20 Hz 的观测和五个轮换时序槽匹配 4 Hz 训练采样，收齐 1.5 秒历史后才请求规划。

从仓库根目录启动推理服务。若需要迁移 VLM 路径，指向训练使用的相同模型和 tokenizer。

```bash
python -m unidrivevla.closedloop.lead_server \
  --checkpoint /path/to/lead/last.ckpt --device cuda \
  --vlm-path /path/to/trained-vlm
```

新建前端配置文件。

```yaml
server_url: http://127.0.0.1:9001
timeout: 60
```

启动 CARLA 和 traffic manager 后，从 `unidrivevla/` 包目录运行。

```bash
LEAD_ROOT=/path/to/pinned-lead \
CARLA_ROOT=/path/to/carla \
ROUTES=/path/to/official-route.xml \
LEAD_AGENT_CONFIG=/path/to/lead-agent.yaml \
OUTPUT_DIR=/path/to/new-lead-evaluation \
LEAD_PYTHON=/path/to/official-lead-env/bin/python \
LEAD_MODE=standard bash scripts/infer/run_lead.sh
```

使用 Bench2Drive 路线和 devkit 时选 `LEAD_MODE=bench2drive`。评测要求新建输出目录。每个评测进程使用独立的模型服务端口，防止不同路线共享时序状态。真实 CARLA rollout 和官方分数待验收。
