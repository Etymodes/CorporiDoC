# CorporiDoC

CorporiDoC is a research-oriented desktop application for markerless pose estimation,
video-derived motor phenotyping, and clinician-supervised assessment support in patients
with disorders of consciousness (DoC).

Joint development context: Tiantan Hospital DoC team (**TiantanDoC**).

## Status

Early research prototype. **Not a medical device and not an autonomous diagnostic system.**
All machine-generated findings must remain traceable to source video and be reviewed by
qualified clinicians.

## Planned runtime

- Conda environment: `soma`
- Python: 3.10
- Desktop UI: PySide6
- Pose backends: pluggable; DeepLabCut is the first custom-model backend
- Local metadata store: SQLite
- Video and derived artifacts: local project workspace, never committed to Git

## Development rule

Each runnable increment is developed on a short-lived branch, validated, committed, and
reviewed as a small pull request. See `docs/TASKBOOK.md` after Milestone 1 lands.

## References

- Mathis et al. (2018), DeepLabCut: https://doi.org/10.1038/s41593-018-0209-y
- Nath et al. (2019), DeepLabCut workflow: https://doi.org/10.1038/s41596-019-0176-0
- Kane et al. (2020), DeepLabCut-Live!: https://doi.org/10.7554/eLife.61909
- Giacino et al. (2004), CRS-R: https://doi.org/10.1016/j.apmr.2004.02.033
