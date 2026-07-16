# CorporiDoC × TiantanDoC 开发任务书

版本：0.8.0（Milestone 5）  
运行环境：`soma`（Python 3.10）  
仓库：`Etymodes/CorporiDoC`

## 1. 项目定义

CorporiDoC 是针对灵长类、重点面向人类意识障碍患者的视频行为计算平台。它的工程目标不是
把一个动物追踪工具换皮，而是把“患者—采集会话—刺激事件—姿态/面部/眼动推理—人工复核—
临床证据—报告”做成可追溯工作流。

平台服务五个层级：

1. **采集与时间层**：管理视频、EEG、刺激事件、同步锚点、原始时间基和对齐质量；
2. **标注与训练层**：通用人体模型自动粗标、短视频点传播、人工修正、数据集快照和专病训练；
3. **行为量化层**：从关键点序列产生位移、速度、关节角、左右不对称、动作起始、持续时间、
   重复性、刺激锁定反应和微小面部运动等指标；
4. **运动表型研究层**：记录患者纵向康复变化、运动模式、模型实验和可复现实验数据；
5. **临床证据层**：将算法发现作为 CRS-R、MBT-r 及扩展行为观察的候选证据，交由临床人员
   回看视频、确认或驳回，不直接自动判定 UWS/VS、MCS-/MCS+ 或 EMCS。

## 2. 科学与工程依据

### 2.1 从 DLC 继承什么

DeepLabCut 的真正优势是“用户定义关键点 + 迁移学习 + 少量人工标注 + 主动迭代”，而非某一种
固定动物骨架。原始工作表明约 200 个精心抽取和标注的帧即可在多个任务达到接近人工的精度，
但这不是对任意临床分布都成立的固定样本量承诺。

CorporiDoC 借鉴并封装以下 DLC 工作流：

- 项目与视频管理；
- 抽帧、人工标注、训练集生成、训练与评估；
- `analyze_videos` 式批量推理；
- 低置信度帧过滤和异常帧再标注；
- `create_labeled_video` 式标记视频；
- `plot_trajectories`/结构化关键点表式轨迹输出；
- DLC-Live! 的单帧推理/处理器思想，用于未来在线检测。

不复制 DLC 内部实现；先定义 CorporiDoC 自己的 `PoseBackend` 协议，再通过适配器调用 DLC。
这样 DLC 3.x 的 PyTorch 后端升级不会穿透到患者、报告和审计模块。

### 2.2 为什么人体不能只靠 DLC Model Zoo

DLC 3.0 官方基础模型目前重点是 SuperAnimal-Quadruped 和 SuperAnimal-TopViewMouse。人类 DoC
视频具有卧床、遮挡、被褥、管路、护理人员进入画面、极小幅运动和非标准机位等显著域偏移。
因此路线是：

1. 用成熟人体预训练姿态模型快速得到粗骨架和伪标签；
2. 由临床/标注员修正，建立 TiantanDoC 数据集；
3. 用 DLC 自定义患者关键点模型补足眼睑、口角、指尖、足趾和病床参照点；
4. 用患者级拆分验证，禁止同一患者的相邻视频跨训练集与测试集；
5. 对置信度、遮挡、亚组和机位分别校准，而不是只报全体平均像素误差。

### 2.3 与 DoC 临床评估的关系

CRS-R 是行为学诊断的核心工具，但患者反应波动、感觉/运动障碍、镇静、疲劳和检查时机都会造成
假阴性。AAN/ACRM/NIDILRR 指南强调标准化、重复评估，并在适当场景结合其他客观方法。
CorporiDoC 的价值是延长观察窗口、量化微弱且不稳定的反应、保留源证据和跨次比较，不能取代
床旁检查。

优先数字化的候选行为包括：

- 指令后口角、眼睑、下颌或头部的低幅度时锁运动；
- 视觉追踪、视觉定向和注视保持；
- 疼痛定位与撤回模式的区分；
- 自动运动反应、抓握/触物、腿交叉、抗睁眼等扩展行为；
- 左右肢体自主活动、周期性/刻板活动和运动量昼夜变化；
- 呼吸、眨眼和面部动作的变化，但需排除反射、痉挛和护理干预。

2025 年 SeeMe 研究已证明计算机视觉可以检出临床肉眼难以识别的指令相关低幅度面部运动，
这为“刺激事件对齐 + 面部位移场 + 临床结局验证”提供了直接先例，但不能未经外部验证照搬其
阈值或结论。

### 2.4 DLC 核心拆解与多后端路线

DLC 不是一个必须整体嵌入的单模型。对 CorporiDoC 有价值的核心可拆成六个独立环节：

1. 从视频按姿态、机位和困难病例抽取代表帧；
2. 定义研究问题所需关键点并人工标注；
3. 以预训练视觉骨干和关键点定位头进行迁移学习；
4. 逐帧输出坐标和似然/置信度；
5. 过滤低置信度、跳变和失败帧，进入主动学习复核；
6. 冻结数据集快照与模型版本后重训、评估和批量推理。

CorporiDoC 自己保存患者、时间轴、标注版本、模型与产物来源；外部算法只通过适配器读写统一
关键点协议。这样可以在同一实验中组合以下能力，而不让某个框架控制临床数据模型：

| 层级 | 候选方法 | 在本项目中的职责 | 接入优先级与边界 |
|---|---|---|---|
| 通用人体自动粗标 | MediaPipe、MMPose RTMPose/RTMW | 快速得到人体/whole-body 伪标签与置信度 | MediaPipe 已接入；MMPose 在独立环境验证后接入 |
| 短视频半自动传播 | OpenCV LK 光流、TAPIR/TAPNext | 人工点一次后向前后传播，遮挡后提示重标 | 先做轻量 LK；TAP 作为更强可选后端 |
| 专病自定义训练 | DLC 3、Lightning Pose、MMPose 自定义配置 | 学习卧床、被褥、细微末端/面部点 | DLC 为首个训练闭环；Lightning Pose 训练需 Linux/NVIDIA |
| 时序修正与表征 | EKS、MotionBERT | 平滑、3D 提升和运动片段表征 | 必须保留原始点；模型需在 DoC 域重新验证 |
| 动作/运动模式 | 可解释轨迹指标、MMAction2 PoseC3D/ST-GCN | 研究动作片段、运动模式与纵向康复 | 先做可解释指标，专病标签成熟后再分类 |

TAPIR 是“任意点跟踪器”，不是语义人体姿态模型；传播结果必须标记为 `propagated` 并经人工
确认。MotionBERT 和通用动作识别数据主要来自站立/日常动作，不得直接把其类别解释为 DoC
行为。MMPose、TAP、MMAction2 和 MotionBERT 主代码为 Apache-2.0，Lightning Pose 为 MIT；
DLC 主代码为 LGPL-3.0，但每个权重和数据集仍需单独登记许可。

### 2.5 Video-EEG 对齐路线

Video-EEG 分为采集时同步和既有文件离线对齐两条路径：

- **新采集**：优先用 LSL 发送 EEG、刺激/指令、相机或采集程序标记，由 LabRecorder 保存 XDF；
- **既有资料**：读取视频真实 PTS、EEG 采样时间和事件，以 TTL、光电二极管、可见闪光、同步音
  或人工确认事件建立至少两个锚点；
- **统一映射**：保存 `t_eeg = slope * t_video + offset`、锚点来源、RMS/最大残差与算法版本；
- **质量控制**：文件起点、名义 FPS 或单一锚点只可作为低可信提示，不能用于毫秒级结论；
- **分析**：对齐通过后才能计算刺激锁定运动、EEG epoch、反应延迟与跨模态一致性。

LSL/XDF 会保存样本时间戳和时钟偏移，官方导入器使用整段记录估计线性时钟漂移。CorporiDoC
的仿射对齐核心也覆盖非 LSL 资料，但后续要加入稳健拟合、离群锚点提示和设备级同步误差阈值。

## 3. 用户与权限

首期角色：

| 角色 | 允许行为 |
|---|---|
| 临床管理员 | 患者管理、协议配置、最终确认、报告签署、权限分配 |
| 临床评估员 | 发起会话、人工评分、证据确认/驳回、报告草稿 |
| 算法研究员 | 去标识化数据集、模型训练、指标分析，不查看直接身份信息 |
| 标注员 | 分配到的帧与关键点，不访问完整病历 |
| 审计/只读 | 查看模型、报告和操作记录，不修改临床数据 |

Milestone 1 只有单机研究模式；账号、权限和电子签名在伦理与部署方案明确后实现，不能用一个
“登录框”假装具备医疗信息系统安全性。

## 4. 功能标签页

| 标签页 | 核心功能 | 主要产物 |
|---|---|---|
| 开始 | 当前患者、快捷入口、运行状态、医疗安全提示 | 当前上下文 |
| 患者 | 注册、搜索、切换、修改、就诊/会话关联 | 患者研究编号与审计记录 |
| 视频 | 导入、哈希、元数据、机位、会话、刺激事件、质量检查 | 不可变源视频登记 |
| 姿态 | 后端选择、推理、置信度、人工修正、版本比较 | 关键点表、标记视频 |
| 评估 | 指令/刺激对齐、候选反应、人工判定、量表证据 | 结构化评估记录 |
| 报告 | 勾选分项、合并导出、轨迹、图表、标记视频 | PDF/CSV/JSON/MP4 |
| 设置 | 模型、关键点规范、设备、目录、审计和隐私策略 | 可复现配置 |

## 5. 关键点体系（初稿）

采用分层关键点，防止一个巨大骨架让所有任务都变慢：

### 5.1 粗人体骨架

鼻、双眼、双耳、双肩、双肘、双腕、双髋、双膝、双踝；用于全身活动、侧别、姿态和裁剪。

### 5.2 DoC 细粒度肢体点

双手掌中心、拇指/食指/小指指尖、双足跟、第一/第五趾、床面参照点；用于抓握、触物、撤回、
定位和小幅末端运动。

### 5.3 面部与眼部

上下眼睑、内外眦、瞳孔中心/边界、双口角、上下唇中点、鼻翼、下颌点；用于眨眼、眼动、口部
指令和 SeeMe 类微运动。面部模型与粗骨架分开运行，并保存人脸 ROI 追踪置信度。

### 5.4 场景与干扰物

床角/床栏、刺激物、镜子/球、评估员手部，以及管路/被褥遮挡掩膜。场景点不与人体点混为一个
临床骨架，但需要共享同一时间轴。

## 6. 数据与可追溯性

核心实体：

```text
Patient -> Encounter -> RecordingSession -> VideoAsset
                                |-> EEGAsset
                                |-> StimulusEvent
                                |-> Timebase -> SyncAnchor -> AlignmentRun
VideoAsset -> InferenceRun -> KeypointSeries -> ReviewEvent
RecordingSession -> Assessment -> EvidenceItem -> Report
DatasetSnapshot -> TrainingRun -> ModelVersion -> InferenceRun
```

最低数据要求：

- 患者采用去标识化 `patient_code`；姓名/联系方式不应进入算法工作区；
- 源视频登记 SHA-256、拍摄时间、帧率、分辨率、机位、侧别和采集协议；
- 关键点逐行保存 `frame_index`、真实 PTS、坐标空间、点规范、来源类型、来源运行、遮挡与复核状态；
- `manual`、`model`、`pseudo_label`、`propagated` 和 `corrected` 互不冒充，人工修正不覆盖原版本；
- 推理保存模型名称、权重哈希、代码版本、参数、硬件和运行时间；
- 训练保存数据集快照哈希、患者级划分、标注规范、随机种子、配置、依赖与全部评估结果；
- EEG 保存原文件哈希、通道/采样信息和时间基；对齐保存全部锚点、斜率、偏移、残差和质量结论；
- 人工改点不覆盖原始推理，保存修改者、时间、原因和前后值；
- 报告中的每个算法结论可跳回时间段、源帧和模型版本；
- Git 忽略数据库、视频、模型和导出物；正式部署采用加密存储、备份和访问审计。

## 7. 输出设计

报告生成器采用“模块清单 + 依赖检查”：

- 患者/会话摘要；
- 视频质量与有效分析时长；
- 全身姿态质量；
- 肢体运动量和左右不对称；
- 关键动作片段；
- 视觉/面部/指令相关候选反应；
- 人工确认的 CRS-R/MBT-r 证据；
- 轨迹图、关节角/速度时序、事件对齐图；
- 模型、阈值、失败帧比例与限制；
- 临床人员签署区。

可选产物：

- 标记视频：原视频叠加骨架、置信度、事件和人工确认标志；
- 轨迹视频：保留时间衰减尾迹、运动方向和左右配色；
- CSV/Parquet：逐帧关键点和派生指标；
- JSON：FHIR 对接前的内部结构化交换格式；
- PDF：固定版临床研究报告。

任何“合并报告”只引用既有版本化产物，不在导出时偷偷重新推理。

## 8. 里程碑与小 Demo 提交

每个里程碑至少一个可运行 Demo；一个提交只回答一个清晰问题。

### M0：仓库初始化（已完成）

- 项目定位、医疗边界、运行环境和核心参考；
- 提交：`Initialize CorporiDoC`。

### M1：患者管理与应用壳（已完成）

交付：开始界面、标签页、患者注册/切换/修改、SQLite、审计事件、单元测试。  
验收：使用虚构患者可完成增改查；重复编号被拒绝；重启后仍可读取；测试通过。  
提交：`Add patient registry demo`。

### M2：视频导入与质控

交付：文件选择、SHA-256、OpenCV/ffprobe 元数据、缩略图、播放、会话绑定、质量警告。  
验收：同一文件不重复入库；损坏视频有明确错误；不改写源文件；可登记机位和拍摄协议。  
建议拆分：`Add immutable video intake`、`Add video quality summary`。

M2A 当前交付：按患者登记源视频路径、将视频复制到患者应用目录、复制前后 SHA-256 校验与
内容去重、OpenCV 基础元数据、首帧可解码验证、原路径/应用副本缺失提示和导入审计。应用副本
按患者数据库 ID 和内容哈希寻址，患者改编号或原路径移动不会破坏记录。误登记可删除数据库记录
和应用副本，原视频不受影响，并保留删除审计事件。OpenCV 返回值暂不当作精确采集时钟；可变
帧率、编码和音视频流信息将在 M2B 使用 `ffprobe` 复核。

M2B 第一小 Demo：导入前确认机位、观察侧别、采集协议和视频备注；按 `m2b-basic-v1` 保存
基础元数据质控。当前工程警告覆盖低于 640×480、低于 15 FPS，以及帧率、帧数或时长缺失。
这些阈值只用于提示人工复核，不代表视频达到临床分析要求；已有视频显示“未评估”，不伪造
历史质控结果。后续小 Demo 再加入 `ffprobe`、缩略图和播放。

M2B 第二小 Demo：双击视频行或使用详情按钮查看完整登记信息，并只允许修改机位、观察侧别、
采集协议和备注。患者归属、路径、SHA-256、导入时间和质控结果保持不可变；无变化不产生审计，
实际修改写入 `UPDATE_VIDEO_METADATA`，且审计摘要只记录变化字段名，不复制备注正文。

M2B 第三小 Demo：在应用内播放所选视频，提供播放/暂停、停止、静音和毫秒级进度定位。播放源
优先使用患者目录管理副本，缺失时回退原路径；两条路径都缺失则拒绝播放。播放器不修改视频，
并将 Qt 多媒体后端的解码错误呈现给用户。详情编辑的 Mac GUI 复核按用户要求暂缓补做。

### M3：姿态后端协议与 Mock 推理

交付：`PoseBackend`、`InferenceRequest/Result`、作业状态、取消和日志；Mock 后端生成可验证轨迹。  
验收：UI 不依赖 DLC 也能完整跑通任务生命周期；失败/取消不会产生“完成”报告。  
提交：`Add pose backend contract`、`Add mock inference demo`。

M3 第一小 Demo：定义不依赖 Qt、DeepLabCut 或具体人体模型的 `PoseBackend` 协议；推理请求固定
患者、视频哈希、后端/模型版本、参数和所需产物，结果只接受成功、失败或取消终态。进度回调
使用帧数表达，取消令牌可由执行器和后端共同检查。此提交只建立边界，不创建任务队列、不运行
模型；Mock 后端和任务界面分别在后续小 Demo 接入。

M3 第二小 Demo：确定性 Mock 后端读取实际视频帧，生成鼻尖和双腕三点的虚拟轨迹 CSV，所有行
均标记 `mock-not-clinical`。运行前复核视频 SHA-256，输出采用临时文件完成后原子改名；失败或
取消不登记产物并清理临时文件。该后端只验证软件任务生命周期，不表示任何患者姿态或临床事实。

M3 第三小 Demo：“姿态”标签页按当前患者列出视频，在 `QThread` 后台运行 Mock 后端，并显示
帧进度、成功/失败/取消终态、产物路径和 SHA-256。任务启动后固定输入视频，患者切换不篡改
运行中任务；关闭应用时先请求取消，避免工作线程被直接销毁。推理运行记录和审计将在下一小
Demo 落库，本提交只验证 UI 生命周期。

M3 第四小 Demo：SQLite 新增 `inference_runs` 和 `inference_artifacts`，记录患者/视频、输入哈希、
后端与模型版本、请求参数、时间、帧数、警告、错误、终态及产物哈希。启动和结束分别写入审计；
只有成功任务可登记完成产物，文件必须位于应用数据目录且通过 SHA-256 复核。此提交先完成仓储
接口，下一小 Demo 再与姿态页接线。

M3 第五小 Demo：姿态页在启动线程前登记运行记录，收到终态后复核并登记产物，同时显示当前
患者的历史任务。结果无法安全落库时界面按失败处理，不把孤立文件显示为成功证据；应用异常退出
留下的 `running` 记录在下次单机启动时转为失败并写入恢复审计。首期仍假设同一数据库只有一个
CorporiDoC 进程使用，多工作站任务租约将在部署里程碑实现。

### M4：首个真实人体姿态 Demo

交付：人体预训练模型适配器、单人检测、关键点 CSV、叠加视频。  
验收：对公开/获授权示例视频运行；显示逐点置信度；报告失败帧与遮挡；禁止使用真实患者视频入 Git。  
提交：`Add human pose baseline`、`Export labeled pose video`。

M4 第一小 Demo：采用 MediaPipe Pose Landmarker 作为 Apple Silicon 可运行的人体粗骨架基线，
优先使用 full 模型；不由应用联网下载权重。用户选择的 `.task` 文件复制到受管 `models` 目录，
以内容哈希命名并在复制后复核 SHA-256。SQLite 登记模型名称、后端、版本、文件大小、哈希、
许可证和来源网址，导入动作写入审计。预检同时验证模型文件和 `mediapipe` Python 包，并固定
显示“未针对 DoC 患者完成临床验证”的警告。下一小 Demo 增加设置页导入入口，再接真实逐帧推理。

M4 第二小 Demo：“设置”标签页可选择本地 `.task` 文件并填写名称、版本、许可证和来源网址；
默认提供官方 full 模型地址与模型卡所列 Apache-2.0 许可证。导入后显示短哈希、时间和实时预检
状态。预检失败不会删除模型登记，便于区分“依赖未安装”和“模型文件损坏”；下一小 Demo 由
姿态页选择已登记且预检通过的模型运行真实关键点导出。

M4 第三小 Demo：实现 `MediaPipePoseBackend`，使用官方 VIDEO 模式和单人设置逐帧推理。关键点
CSV 固定为 33 点规范，保存帧号、毫秒时间戳、归一化坐标、像素坐标、visibility、presence 和
以髋中点为原点的米制世界坐标。未检出帧仍写入 33 行空坐标并标记 `detected=0`；低可见度点、
缺少世界坐标、提前解码结束均进入运行警告。视频和模型在运行前再次校验 SHA-256，取消或失败
清除半成品。下一小 Demo 把该后端接入姿态页并显示警告，再生成骨架叠加视频。

M4 第四小 Demo：“姿态”页现在可在 Mock 与已登记 MediaPipe 模型之间切换。真实任务启动前
必须通过依赖和权重预检，并将三个置信度阈值随推理请求落库；后台线程持有任务启动时选定的后端，
患者或设置变化不会篡改运行中输入。完成界面显示所有运行警告与产物哈希，不把“任务技术完成”
写成“患者姿态正常”。设置页导入模型后会刷新后端列表。下一小 Demo 增加骨架叠加视频。

M4 第五小 Demo：MediaPipe 任务可勾选在同一次逐帧推理中生成骨架叠加 MP4，避免为可视化重复
运行模型。叠加仅绘制 visibility 和 presence 均不低于 0.5 且位于画面内的点；未检出帧保留原
画面并标记 `NO POSE DETECTED`。每帧写入模型短哈希与 `NOT CLINICALLY VALIDATED`，MP4 与 CSV
均经 SHA-256 登记；任一编码步骤失败或任务取消都会清理半成品。至此形成 M4 首个 Mac 集中验收
点。当前标记视频由 OpenCV 重编码为无声恒定帧率 MP4，源视频保持不变；保留音频/原时间基将在
后续 ffmpeg 导出器实现。验收时安装 ARM64 MediaPipe、导入官方 full 模型、运行公开/虚构示例
视频并检查 CSV 与标记视频。

### M5：统一时间基与关键点数据规范（进行中）

交付：视频真实 PTS、通用 `KeypointSeries`、同步锚点、视频↔EEG 时间映射和对齐质控。  
验收：不以 OpenCV 名义 FPS 代替真实 PTS；至少两个锚点；保存斜率、偏移、RMS/最大残差；
任意后端结果可映射到相同的患者/会话时间轴。  
建议提交：`Add video EEG time alignment`、`Add canonical keypoint series`、`Probe exact video PTS`。

M5 第一小 Demo：标准库实现两个或更多同步锚点的仿射最小二乘拟合，支持视频时间与 EEG 时间
双向换算，并返回锚点数、RMS 和最大残差。锚点必须在两条时间轴上严格递增；当前不自动删除
离群点，也不判定“临床同步合格”。下一小 Demo 将模型与锚点持久化，再接 `ffprobe` 真实 PTS。

### M6：短视频半自动标注

交付：选点、短区间前后传播、遮挡/漂移提示、逐点接受/拒绝和来源记录。  
步骤：先复用 OpenCV LK 光流完成轻量 Demo；再以独立可选环境评估 TAPIR/TAPNext；最后比较
每分钟人工修正数、遮挡恢复率和端点误差。  
验收：传播点标记 `propagated`；失败后停止或要求重锚定；不覆盖人工点；同一界面可对比通用
姿态粗标与点传播结果。

### M7：人工复核、轨迹与数据集快照

交付：低置信度/跳变/缺失帧队列、逐点修正、版本比较、轨迹图/视频、冻结数据集快照。  
验收：修改可撤销且有审计；插值与伪标签不冒充人工标注；轨迹使用真实时间戳；训练集按患者
划分并记录来源比例。  
建议提交：`Add keypoint review queue`、`Add trajectory exports`、`Freeze dataset snapshots`。

### M8：TiantanDoC 自定义模型训练闭环

交付：DLC 项目导入/创建、抽帧、标注转换、训练配置、训练运行、评估、模型登记与批量推理。  
第二候选为 MMPose 自定义 whole-body；Lightning Pose 用于服务器端时序/半监督实验。  
验收：不复制外部源码；训练与桌面推理解耦；记录代码/权重/配置/硬件/随机种子；比较 MediaPipe、
MMPose 与专病模型；可从冻结快照重现。  
建议提交：`Add DeepLabCut project adapter`、`Register training runs`、`Compare pose backends`。

### M9：运动表型与康复纵向研究

交付：有效检测时长、路径长度、幅度、速度、加速度、平滑度/jerk、关节角、左右对称、运动 bout、
反应延迟和重复性；按患者会话显示基线与变化。  
验收：指标单位和归一化规则固定；缺失/遮挡不会被当作静止；报告测试—重测可靠性和最小可检测
变化；支持导出实验数据与模型输入。  
通用动作分类不先行，先建立可解释、可复核的专病运动表型。

### M10：刺激事件与 Video-EEG

交付：EEG/XDF 导入、通道与采样元数据、LSL/TTL/光电/音视频锚点、刺激时间线、漂移与残差 QC、
运动事件和 EEG epoch 联合浏览。  
验收：事件对齐误差有测量；护理操作可标为混杂；原始 EEG 不被改写；可从对齐事件回放视频并
定位 EEG 区间；失败对齐不得生成时锁结论。  
建议提交：`Import EEG recordings`、`Persist synchronization anchors`、`Align video EEG events`。

### M11：面部/眼动与专病运动模式模型

交付：面部 ROI 配准、眨眼/口角/下颌、视觉追踪、扩展行为候选，以及在专病标签成熟后的
PoseC3D/ST-GCN 或 MotionBERT 表征实验。  
验收：对头动和相机抖动做配准；患者外测试；区分自发、护理诱发与指令相关；每个候选动作可
回放；禁止一次提交“全套意识识别”。

### M12：临床证据与模块化报告

交付：CRS-R/MBT-r 录入、候选证据确认/驳回、勾选模块、PDF/CSV/JSON/MP4 导出。  
验收：不由总分简单替代诊断规则；报告标出评估者、版本、同步质量、模型限制和未完成模块；
合并报告只引用已冻结产物，不重新推理。

### M13：安全部署与临床研究验证

交付：角色权限、电子签名、迁移、加密、备份恢复、脱敏导出、冻结研究协议、样本量估算、患者级
外部验证、亚组/机位/病因分析和失败案例库。  
验收：威胁建模、最小权限、恢复演练、真实数据与开发仓库隔离；主要终点预注册；医院信息部门、
伦理和法规共同确认。在此之前不得宣传“实现自动诊断”。

## 9. 模型评估方案

### 9.1 姿态层

- 患者级 train/validation/test 划分；
- 标准化关键点误差（按头宽、肩宽或肢段长度）；
- PCK、OKS/mAP、每关键点误差分布、置信度校准；
- 遮挡、机位、光照、被褥、病因和诊断亚组分层；
- 视频级失败率与连续丢点长度，不能只算可见帧。

### 9.2 事件层

- 事件起始时间误差、持续时间误差；
- 灵敏度、特异度、PPV、NPV、F1 与每小时误报；
- 受试者外验证和重复会话稳定性；
- 与至少两名盲法临床评估员比较，报告 Cohen/Fleiss kappa 或 ICC。

### 9.3 临床层

- 对 CRS-R/MBT-r 的增量证据，而非训练集上重建量表总分；
- 预先规定参考标准、复测次数、药物/觉醒周期和感觉运动障碍；
- 对假阳性和假阴性分别做临床风险分析；
- 模型沉默/拒绝输出是合法结果，低质量视频不得强行分类。

## 10. 工程完成定义（Definition of Done）

每个小 Demo 合并前必须：

1. 能在 `soma` 环境从干净安装启动；
2. 至少包含正常路径和一个失败路径测试；
3. 不包含患者数据、视频、权重、密钥和本机绝对路径；
4. UI 中区分“算法候选”“人工确认”“临床结论”；
5. 新生成物有来源、参数和版本；
6. 更新任务书或 README 中对应状态；
7. 一个语义清晰的提交，先开草稿 PR 再合并。

## 11. 当前已知风险与决策

| 风险 | 当前决策 |
|---|---|
| DLC 对人体临床域并无开箱即用基础模型 | 多后端粗定位 + TiantanDoC 自定义 DLC 细点模型 |
| 低幅运动接近压缩/相机抖动噪声 | 固定机位、场景参照点、视频质控、配准和重复试次 |
| 同一患者相邻帧造成数据泄漏 | 患者级拆分，必要时按中心/设备做外部测试 |
| 自动诊断带来医疗风险 | 只输出可回放候选证据，临床确认后才进入报告 |
| 真实患者隐私 | 去标识化、开发数据隔离、Git 强制忽略、正式安全评审 |
| DLC LGPL 与 SuperAnimal 权重限制 | 适配器调用、不复制源码；模型许可证逐一登记 |
| 通用人体/动作模型与卧床 DoC 域偏移 | 只作粗标或研究基线；TiantanDoC 数据患者级外部验证 |
| 点传播在遮挡和长视频中漂移 | 限制短区间、双向检查、重锚定并保留 `propagated` 来源 |
| Video 与 EEG 时钟偏移/漂移 | 多锚点仿射映射、残差质控；拒绝只按文件起点对齐 |
| Lightning Pose 和 OpenMMLab 对 Mac 依赖较重 | 桌面只保留适配协议；训练放隔离 Linux/GPU 环境 |
| 公开仓库与联合开发知识产权 | 正式 LICENSE、署名、数据权属由双方在发布前书面确定 |

## 12. 主要文献与源码

- Mathis A, et al. DeepLabCut: markerless pose estimation of user-defined body parts with deep
  learning. *Nature Neuroscience*. 2018. https://doi.org/10.1038/s41593-018-0209-y
- Nath T, et al. Using DeepLabCut for 3D markerless pose estimation across species and behaviors.
  *Nature Protocols*. 2019. https://doi.org/10.1038/s41596-019-0176-0
- Kane GA, et al. Real-time, low-latency closed-loop feedback using markerless posture tracking.
  *eLife*. 2020. https://doi.org/10.7554/eLife.61909
- Lauer J, et al. Multi-animal pose estimation, identification and tracking with DeepLabCut.
  *Nature Methods*. 2022. https://doi.org/10.1038/s41592-022-01443-0
- Ye S, et al. SuperAnimal pretrained pose estimation models for behavioral analysis.
  *Nature Communications*. 2024. https://doi.org/10.1038/s41467-024-48792-2
- Bala PC, et al. Automated markerless pose estimation in freely moving macaques with
  OpenMonkeyStudio. *Nature Communications*. 2020. https://doi.org/10.1038/s41467-020-18441-5
- Giacino JT, et al. The JFK Coma Recovery Scale-Revised: measurement characteristics and
  diagnostic utility. 2004. https://doi.org/10.1016/j.apmr.2004.02.033
- Giacino JT, et al. Practice guideline update recommendations summary: Disorders of
  consciousness. *Neurology*. 2018. https://doi.org/10.1212/WNL.0000000000005926
- Pincherle A, et al. Motor behavior unmasks residual cognition in disorders of consciousness.
  *Annals of Neurology*. 2019. https://doi.org/10.1002/ana.25417
- Hoyoux T, et al. Computer vision assessment of visual pursuit with a moving mirror.
  WACV 2016. https://doi.org/10.1109/WACV.2016.7477604
- Cheng X, et al. Computer vision detects covert voluntary facial movements in unresponsive
  brain injury patients. *Communications Medicine*. 2025.
  https://doi.org/10.1038/s43856-025-01042-y
- DeepLabCut source: https://github.com/DeepLabCut/DeepLabCut
- DeepLabCut-Live! source: https://github.com/DeepLabCut/DeepLabCut-live
- DeepLabStream source: https://github.com/SchwarzNeuroconLab/DeepLabStream
- SLEAP source: https://github.com/talmolab/sleap
- MMPose/RTMPose source: https://github.com/open-mmlab/mmpose
- RTMPose paper: https://arxiv.org/abs/2303.07399
- Biderman D, et al. Lightning Pose: improved animal pose estimation via semi-supervised
  learning, Bayesian ensembling and cloud-native open-source tools. *Nature Methods*. 2024.
  https://doi.org/10.1038/s41592-024-02319-1
- Lightning Pose source: https://github.com/paninski-lab/lightning-pose
- Doersch C, et al. TAPIR: Tracking any point with per-frame initialization and temporal
  refinement. ICCV 2023. https://arxiv.org/abs/2306.08637
- TAP source: https://github.com/google-deepmind/tapnet
- MotionBERT source and ICCV 2023 paper: https://github.com/Walter0807/MotionBERT
- MMAction2 source: https://github.com/open-mmlab/mmaction2
- Lab Streaming Layer source: https://github.com/sccn/liblsl
- LSL time synchronization: https://labstreaminglayer.readthedocs.io/info/time_synchronization.html
- PyXDF source: https://github.com/xdf-modules/pyxdf
- MNE-LSL documentation: https://mne.tools/mne-lsl/stable/index.html
