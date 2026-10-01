# HEI Stack100 Workspace

100 条双臂抓取与三方块堆叠示范。每条重新随机布置红、绿、蓝三个 8 cm 方块，双臂依次抓取和放置，形成无机械臂扶持的稳定塔。示范由可读取仿真物体真值的程序专家生成，不是人工手柄录制，也不是 Diffusion Policy 生成。

数据保存在本项目的 [GitHub Release](https://github.com/YouShouldBetOnMe/hei-mujoco-sim/releases/tag/dataset-stack100-v1)。完整数据以多个独立 ZIP 文件发布，避免让仿真代码仓库携带约 9.9 GiB 数据。校验清单和数据元信息随代码提交；下载后恢复原始目录及文件内容。

## 下载

在仓库根目录使用 Python 3.10 或更新版本。下载器只使用 Python 标准库，无需先安装仿真或训练依赖。

```bash
# 默认下载可用于训练的 LeRobot v3 格式
python scripts/download_dataset.py

# 下载完整原始 PNG、状态、动作和物理轨迹
python scripts/download_dataset.py --format raw

# 同时下载两种格式；从未下载过的目录运行
python scripts/download_dataset.py --format all
```

两种格式分别恢复到 `datasets/hei_stack100_workspace` 和 `datasets/hei_stack100_workspace_raw`。中断后重新运行同一条命令即可继续下载；每卷通过 SHA-256 后才解压，已有完整数据目录不会被覆盖。下载器在 `datasets/.downloads` 保留 ZIP 缓存，确认数据恢复后可以删除这些 ZIP 释放空间。下载、缓存和解压同时存在时需要约两倍对应数据大小的磁盘空间。

也可以手动从 Release 下载某种格式的全部 ZIP 文件，再运行：

```bash
python scripts/download_dataset.py --format lerobot --archive-dir /path/to/downloaded/zips
python scripts/download_dataset.py --format all --archive-dir /path/to/downloaded/zips --verify-only
```

各 ZIP 是独立归档，无需拼接；同一种格式的所有卷须解压到同一父目录。[dataset_release.json](dataset_release.json) 列出完整文件名、下载链接、字节数和 SHA-256。

## 数据规格

| 项目 | 内容 |
|---|---|
| 示范 / 观测帧 | 100 / 51,940 |
| 图像数量 | 155,820 |
| 相机 | `front`、`left_wrist`、`right_wrist`；总览不进入 observation |
| 图像 | 224×224 RGB，三路对应同一观测时刻 |
| 频率 | 10 Hz 仿真时间 |
| 状态 / 动作 | 18 维；双臂关节、夹爪、固定底盘、固定升降 |
| 动作含义 | 绝对位置伺服目标；关节 rad、夹爪总开口 m、升降 m |
| 机体 | 底盘固定；升降 DOF 移除并固定在 -0.25 m |
| 初始方块范围 | 实际中心 x 约 0.502–0.861 m，y 约 -0.519–0.521 m |
| 先行手 | 左手 50 条、右手 50 条 |
| 颜色堆叠顺序 | 全部 6 种排列，分别 16 或 17 条 |
| 数据划分 | 按整条示范划分：训练 90、验证 10 |

原始帧遵循 `observation_t -> action_t -> physics_step`，图像及测量状态来自动作执行前。`object_trace.json` 是动作执行后的辅助物体轨迹，不能当作同一时刻的策略观测。初始布局、朝向和路径都有变化；每条示范使用不同种子与初始状态。

## 两种格式

`hei_stack100_workspace` 包含 LeRobot v3 的 `data/`、`meta/` 和验证报告。三路图像以内嵌字节保存于 Parquet，不需要外部图像下载。该目录约 4.22 GiB。

在兼容 LeRobot v3 的训练环境中读取：

```python
import json
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path('datasets/hei_stack100_workspace')
splits = json.loads((root / 'meta/splits.json').read_text())
train = LeRobotDataset('local/hei_stack100_workspace', root=root,
                       episodes=splits['train'], video_backend='pyav')
validation = LeRobotDataset('local/hei_stack100_workspace', root=root,
                            episodes=splits['validation'], video_backend='pyav')
```

必须应用 `meta/splits.json`。`meta/info.json` 中的默认 `train: 0:100` 表示完整数据可读取，并没有应用本批次的 90/10 留出划分。避免直接把全部 100 条用于训练后又在留出的 10 条上报告验证结果。训练环境需另行安装 LeRobot/PyTorch；本仓库的基础 MuJoCo 安装不包含这些依赖。

`hei_stack100_workspace_raw` 约 5.69 GiB，保留全部原始文件：100 个完整示范目录、三路 PNG、`frames.jsonl`、观测时刻 `physics_trace.npz`、动作阶段、任务成功证据、采集清单及候选物理轨迹。可使用 `hei_sim.recording.validate_episode` 检查每个正式示范目录。`.npz` 中的 qpos/qvel 属于移除升降 DOF 的模型；加载仿真时用 `Environment(fixed_lift=-0.25)`，不能直接配合默认可动升降模型重放。

## 验证与范围

发布前重新解码全部三路 PNG，校验帧数、图像尺寸、时间对齐、状态/动作、物理轨迹、100 个不同种子/初始状态以及每条的连续 3 秒稳定堆叠成功证据。[audit_summary.json](audit_summary.json) 是本次发布审计；[validation_report.json](validation_report.json) 包含原有转换和 Diffusion Policy 输入批次检查。详细配置见 [simulation.json](simulation.json)，训练划分见 [splits.json](splits.json)。

重力、接触和摩擦启用，物体没有在任务执行中被传送或挂接。机器人自身碰撞尚未启用，相机安装和动力学尚未实机标定。这是当前仿真任务的程序专家数据，不代表学得策略成功率、原论文基准成绩或真机迁移效果。

本项目发布的数据按仓库 [Apache 2.0 许可证](../../LICENSE) 提供；图像包含的机器人/YCB 模型资源保留 [NOTICE](../../NOTICE) 中的来源与署名。
