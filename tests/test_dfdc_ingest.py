import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avdf.ingest import build_dfdc, dfdc_status, group_stratified_split  # noqa: E402


def make_dfdc(tmp_path, n_src=12, fakes_per_src=2):
    """n_src REAL source videos, each with fakes_per_src FAKEs; all on disk except src0's REAL."""
    meta = {}
    for s in range(n_src):
        meta[f"src{s}.mp4"] = {"label": "REAL", "split": "train", "original": None}
        for k in range(fakes_per_src):
            meta[f"fk{s}_{k}.mp4"] = {"label": "FAKE", "split": "train", "original": f"src{s}.mp4"}
    meta["gone.mp4"] = {"label": "REAL", "split": "train", "original": None}   # listed but never downloaded
    for name in meta:
        if name not in ("src0.mp4", "gone.mp4"):
            (tmp_path / name).write_bytes(b"")
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    return tmp_path


def test_labels_and_grouping(tmp_path):
    df = build_dfdc(make_dfdc(tmp_path))
    assert len(df) == 12 * 3 - 1                       # src0.mp4 is not on disk
    assert set(df.label) == {"real", "fake"}
    fk = df[df.path.str.endswith("fk3_1.mp4")].iloc[0]
    assert fk.label == "fake" and fk.subject == "src3"
    assert df[df.path.str.endswith("src3.mp4")].iloc[0].subject == "src3"


def test_split_is_leak_free_and_has_both_classes(tmp_path):
    df = group_stratified_split(build_dfdc(make_dfdc(tmp_path)))
    assert df.groupby("subject").split.nunique().max() == 1   # a source never spans two splits
    assert set(df.split) == {"train", "val", "test"}
    assert df[df.split == "train"].label.nunique() == 2


def test_status_lists_missing_real(tmp_path):
    st = dfdc_status(make_dfdc(tmp_path))
    assert (st["have_real"], st["have_fake"]) == (11, 24)
    assert st["real_missing"] == ["gone.mp4", "src0.mp4"]
    assert st["real_missing_paired"] == ["src0.mp4"]       # the original of fakes already on disk
