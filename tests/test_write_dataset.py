from pathlib import Path

from metaorder_signal.synthetic_market import SyntheticMarketConfig, write_synthetic_dataset


def test_write_synthetic_dataset_manifest(tmp_path: Path):
    cfg = SyntheticMarketConfig(n_metaorders=5, n_sessions=1, length_cap=30)
    manifest = write_synthetic_dataset(
        ["AAA", "BBB"],
        tmp_path,
        seed=3,
        cfg=cfg,
        fmt="csv",
    )
    assert manifest["total_rows"] > 0
    mpath = Path(manifest["manifest_path"])
    assert mpath.is_file()
    assert (tmp_path / "AAA.csv").is_file()
    assert (tmp_path / "BBB.csv").is_file()
