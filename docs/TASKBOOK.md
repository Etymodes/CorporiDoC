# CorporiDoC × TiantanDoC 开发任务书

版本：0.2（Milestone 2A）  
运行环境：`soma`（Python 3.10）  
仓库：`Etymodes/CorporiDoC`

## 1. 项目定义

CorporiDoC 是针对灵长类、重点面向人类意识障碍患者的视频行为计算平台。它的工程目标不是
把一个动物追踪工具换皮，而是把“患者—采集会话—刺激事件—姿态/面部/眼动推理—人工复核—
临床证据—报告”做成可追溯工作流。

平台服务三个层级：

1. **自动标记层**：人体和自定义细粒度关键点、标记视频、运动轨迹和质量置信度；
2. **行为量化层**：从关键点序列产生位移、速度、关节角、左右不对称、动作起始、持续时间、
   重复性、刺激锁定反应和微小面部运动等指标；
3. **临床证据层**：将算法发现作为 CRS-R、MBT-r 及扩展行为观察的候选证据，交由临床人员
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
                                |-> StimulusEvent
VideoAsset -> InferenceRun -> KeypointSeries -> ReviewEvent
RecordingSession -> Assessment -> EvidenceItem -> Report
ModelVersion -> InferenceRun
```

最低数据要求：

- 患者采用去标识化 `patient_code`；姓名/联系方式不应进入算法工作区；
- 源视频登记 SHA-256、拍摄时间、帧率、分辨率、机位、侧别和采集协议；
- 推理保存模型名称、权重哈希、代码版本、参数、硬件和运行时间；
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

M2A 当前交付：按患者登记源视频路径、SHA-256 内容去重、OpenCV 基础元数据、首帧可解码
验证、源文件缺失提示和导入审计。M2A 不复制源视频，也不把 OpenCV 返回值当作精确采集时钟；
可变帧率、编码和音视频流信息将在 M2B 使用 `ffprobe` 复核。

### M3：姿态后端协议与 Mock 推理

交付：`PoseBackend`、`InferenceRequest/Result`、作业状态、取消和日志；Mock 后端生成可验证轨迹。  
验收：UI 不依赖 DLC 也能完整跑通任务生命周期；失败/取消不会产生“完成”报告。  
提交：`Add pose backend contract`、`Add mock inference demo`。

### M4：首个真实人体姿态 Demo

交付：人体预训练模型适配器、单人检测、关键点 CSV、叠加视频。  
验收：对公开/获授权示例视频运行；显示逐点置信度；报告失败帧与遮挡；禁止使用真实患者视频入 Git。  
提交：`Add human pose baseline`、`Export labeled pose video`。

### M5：DLC 自定义模型闭环

交付：DLC 项目关联、抽帧/标注入口、训练配置、批量推理、过滤、再标注、模型登记。  
验收：不复制 DLC 源码；记录 DLC/torch/CUDA 版本；训练与推理解耦；可以导入已有 DLC 项目。  
提交：`Add DeepLabCut backend`、`Register model versions`。

### M6：人工复核与轨迹

交付：低置信度帧队列、逐点修正、插值建议、原值/修正值双轨、轨迹图和轨迹视频。  
验收：修改可撤销且有审计；插值不伪装为人工标注；轨迹使用实际时间戳而非假定固定帧率。  
提交：`Add keypoint review queue`、`Add trajectory exports`。

### M7：刺激—反应协议

交付：音频/屏幕/触觉/人工事件时间戳、基线窗/反应窗、重复试次、盲法标记和时锁统计。  
验收：事件对齐误差有测量；护理操作可标成混杂；候选反应必须有可回放证据。  
提交：`Add stimulus event timeline`、`Detect time-locked movement candidates`。

### M8：面部微运动、眼动与扩展行为

交付：面部 ROI、光流/关键点融合、眨眼/口角/下颌、视觉追踪、扩展行为候选。  
验收：对头动、相机抖动做配准；区分自发与指令相关；阈值由开发集确定并锁定到测试集。  
提交按单项拆分，禁止一次提交“全套意识识别”。

### M9：临床证据与模块化报告

交付：CRS-R/MBT-r 录入、候选证据确认/驳回、勾选模块、PDF/CSV/JSON/MP4 导出。  
验收：不由总分简单替代诊断规则；报告标出评估者、版本、限制和未完成模块；可重现。  
提交：`Add clinician evidence review`、`Add modular report export`。

### M10：隐私、安全和部署

交付：角色权限、电子签名、数据库迁移、加密、备份恢复、脱敏导出、日志保留策略。  
验收：威胁建模、最小权限、恢复演练、真实患者数据路径与开发仓库完全分离。  
此阶段需医院信息部门、伦理和法规共同确认。

### M11：临床研究验证

交付：冻结协议、标注手册、样本量估算、患者级外部验证、亚组/机位/病因分析、失败案例库。  
主要终点候选：关键点误差、事件检测灵敏度/特异度、评估者一致性、增量诊断收益、用时。  
在此前不得宣传“实现自动诊断”。

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
