import math

import pytest

from corporidoc.domain import SyncAnchor, SyncMethod, fit_time_alignment


def anchor(video: float, eeg: float, label: str = "flash") -> SyncAnchor:
    return SyncAnchor(video, eeg, SyncMethod.PHOTODIODE, label)


def test_fit_alignment_maps_both_directions() -> None:
    alignment = fit_time_alignment((anchor(0.0, 12.0), anchor(60.0, 72.06)))

    assert alignment.slope == pytest.approx(1.001)
    assert alignment.offset_seconds == pytest.approx(12.0)
    assert alignment.rms_error_seconds == pytest.approx(0.0)
    assert alignment.video_to_eeg(30.0) == pytest.approx(42.03)
    assert alignment.eeg_to_video(42.03) == pytest.approx(30.0)


def test_fit_alignment_reports_residual_quality() -> None:
    alignment = fit_time_alignment(
        (anchor(0.0, 5.0), anchor(10.0, 15.01), anchor(20.0, 25.03))
    )

    assert alignment.anchor_count == 3
    assert alignment.rms_error_seconds > 0
    assert alignment.max_error_seconds >= alignment.rms_error_seconds


@pytest.mark.parametrize(
    "anchors, message",
    [
        ((anchor(0.0, 1.0),), "至少需要两个"),
        ((anchor(0.0, 2.0), anchor(1.0, 1.0)), "EEG.*严格递增"),
    ],
)
def test_fit_alignment_rejects_invalid_anchor_sequences(anchors, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fit_time_alignment(anchors)


def test_anchor_rejects_non_finite_time() -> None:
    with pytest.raises(ValueError, match="有限"):
        anchor(math.nan, 1.0)
