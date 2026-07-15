# CorporiDoC

CorporiDoC 是面向意识障碍（Disorders of Consciousness, DoC）患者的研究型桌面应用，
用于无标记姿态估计、视频运动表型提取和临床人员监督下的评估辅助。联合开发场景为
天坛医院意识障碍团队（TiantanDoC）。

> 当前为研究原型，不是医疗器械，也不能自主作出诊断。所有算法结果必须能回溯到源视频，
> 并由合格临床人员复核。

## Milestone 1

当前小 Demo 已实现：

- 开始界面和功能标签页；
- 注册、切换和修改患者；
- SQLite 本地存储；
- 患者资料创建/修改审计事件；
- 患者数据库、视频、模型和导出物默认不进入 Git。

其余标签页目前是明确标注范围的占位页。

## 在 `soma` 环境运行

```bash
conda activate soma
python --version
python -m pip install -e ".[dev]"
pytest -q
python -m corporidoc
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

