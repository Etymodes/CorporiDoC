# CorporiDoC

CorporiDoC 是面向意识障碍（Disorders of Consciousness, DoC）患者的研究型桌面应用，
用于无标记姿态估计、视频运动表型提取和临床人员监督下的评估辅助。联合开发场景为
天坛医院意识障碍团队（TiantanDoC）。

> 当前为研究原型，不是医疗器械，也不能自主作出诊断。所有算法结果必须能回溯到源视频，
> 并由合格临床人员复核。

## Milestone 4

当前小 Demo 已实现：

- 开始界面和功能标签页；
- 注册、切换和修改患者；
- SQLite 本地存储；
- 患者资料创建/修改审计事件；
- 按当前患者导入视频并建立应用副本；
- 删除误登记视频及其应用副本，同时保留原视频和审计事件；
- 登记机位、观察侧别、采集协议和视频备注；
- 查看完整视频登记详情，并受限修改采集信息；
- 在应用内播放视频，优先使用管理副本并在必要时回退原路径；
- 保存版本化基础质控结果，提示低分辨率、低帧率和元数据缺失；
- SHA-256 内容去重；
- 分辨率、帧率、帧数、时长和文件大小读取；
- 源文件缺失检查和视频导入审计；
- 患者数据库、视频、模型和导出物默认不进入 Git。

M3 已定义与模型框架无关的 `PoseBackend` 协议，以及可追溯的推理请求、终态结果、产物、进度
和取消结构。确定性 Mock 后端可逐帧读取视频并生成明确标注为非临床结果的关键点 CSV，用于在
不安装模型的情况下验证任务生命周期。后续真实人体模型和 DeepLabCut 适配器共享同一接口。

“姿态”标签页现可为当前患者选择视频，在后台运行或取消 Mock 任务，并显示进度、终态、CSV
路径及产物哈希。患者切换不会改变已启动任务的输入；关闭应用时会先取消运行中的任务。

SQLite 现已具备推理运行与产物记录：保存患者/视频、后端/模型版本、视频哈希、参数、终态、
警告、错误和产物哈希，并写入启动/结束审计。姿态页会自动登记任务并显示当前患者的历史；异常
退出遗留的“运行中”记录会在下次启动标记为失败并审计，不会长期伪装成活跃任务。

CorporiDoC 保留原始路径作为来源记录，同时将视频复制到
`~/.corporidoc/patients/patient-XXXXXX/videos/`，复制后复核 SHA-256。原文件不会被修改，
移动原文件也不影响应用副本。

M4 选择 MediaPipe Pose Landmarker 作为首个真实人体姿态工程基线。模型文件由用户明确导入，
复制到 `~/.corporidoc/models/` 后复核 SHA-256，并登记模型名称、版本、后端、许可证和来源网址；
应用不在后台自动下载或静默替换权重。启动真实推理前会同时检查模型文件完整性和 MediaPipe
依赖。该模型主要面向通用人体/健身场景，不能把其输出直接解释为 DoC 临床证据。
“设置”标签页提供模型导入和预检状态；目前尚未从“姿态”页运行该模型。

真实 MediaPipe 后端现可生成版本化 33 点 CSV，并保留未检出帧、逐点 visibility/presence、
归一化/像素坐标和米制世界坐标。该后端将在下一小 Demo 接入“姿态”标签页。

基础质控是工程预警，不是临床可用性结论。OpenCV 返回的 FPS、帧数和时长可能受视频后端
影响；M2B 后续将使用 `ffprobe` 补充可变帧率、编码和音视频流信息。

双击视频行或点击“查看/修改信息”可修改机位、观察侧别、采集协议和备注。患者归属、路径、
SHA-256、导入时间和质控结果保持只读；实际修改会写入 `UPDATE_VIDEO_METADATA` 审计事件。

选择视频后点击“播放所选视频”可在应用内播放、暂停、停止、静音和拖动进度。播放器优先使用
患者目录中的应用副本；仅当副本缺失时回退到原路径，并明确显示当前播放来源。文件存在不代表
编码一定受系统多媒体后端支持，解码失败会在播放器中显示错误。

## 在 `soma` 环境运行

```bash
conda activate soma
python --version
python -m pip install -e ".[dev]"
pytest -q
python -m corporidoc
```

安装 M4 人体姿态可选依赖：

```bash
python -m pip install -e ".[dev,human-pose]"
```

如尚未创建 `soma`：

```bash
conda env create -f environment.yml
conda activate soma
```

指定本地数据目录：

```bash
python -m corporidoc --data-dir /path/to/local/corporidoc-data
```

默认数据库位于 `~/.corporidoc/corporidoc.sqlite3`。请勿将真实患者数据放入 Git 仓库。

## 为什么不把 DeepLabCut 源码复制进来

CorporiDoC 将 DeepLabCut（DLC）接入为可替换的姿态后端，通过稳定接口调用训练、推理、
过滤、标记视频和轨迹导出能力。这样既保留 DLC 对自定义关键点与小样本迁移学习的优势，
也允许日后并列接入人体预训练模型、眼动/面部模型和轻量实时推理引擎。

DLC 主项目采用 LGPL-3.0-or-later；CorporiDoC 在正式确定自身许可证前不复制或修改 DLC
源代码。DLC SuperAnimal 权重还带有研究用途限制，且当前官方基础模型主要面向四足动物和
俯视小鼠，不能把它们直接当成人体临床模型。

## 任务书

完整范围、里程碑、临床边界和每个小 Demo 的提交标准见
[`docs/TASKBOOK.md`](docs/TASKBOOK.md)。

## 核心参考

- Mathis et al. (2018), DeepLabCut: https://doi.org/10.1038/s41593-018-0209-y
- Nath et al. (2019), DLC workflow: https://doi.org/10.1038/s41596-019-0176-0
- Kane et al. (2020), DeepLabCut-Live!: https://doi.org/10.7554/eLife.61909
- Giacino et al. (2004), CRS-R: https://doi.org/10.1016/j.apmr.2004.02.033
- MediaPipe Pose Landmarker Python guide:
  https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python
- MediaPipe repository and Apache-2.0 license: https://github.com/google-ai-edge/mediapipe
