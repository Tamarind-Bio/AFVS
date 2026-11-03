"""One end-to-end local screen with the real qvina02 from tools/bin.

The bundled binary is x86-64 Linux, so this runs where that executes (the
docker/Dockerfile.local image, or any x86-64 Linux box); elsewhere it skips."""

import platform
import subprocess
import tarfile

import pandas as pd
import pytest

from conftest import COLLECTIONS, REPO, Screen, ligand_names

QVINA = REPO / "tools" / "bin" / "qvina02"


def qvina_runs():
    if platform.system() != "Linux":
        return False
    try:
        return subprocess.run([str(QVINA), "--version"], capture_output=True, timeout=60).returncode == 0
    except OSError:
        return False


@pytest.mark.skipif(not qvina_runs(), reason="the bundled qvina02 (x86-64 Linux) cannot run here")
def test_real_qvina02_screen(tmp_path):
    screen = Screen(tmp_path, tools_path=REPO / "tools" / "bin", workers=2, threads=1)
    screen.prepare()
    screen.run("afvs_run_local.py", timeout=1800)

    assert screen.done_markers() == ["1/0.done", "2/0.done"]

    scores = pd.concat(pd.read_parquet(p) for p in sorted((screen.job_dir / "qvina02" / "parquet").rglob("*.parquet")))
    expected = sorted(name for key in COLLECTIONS for name in ligand_names(key))
    assert sorted(scores['ligand']) == expected
    # Real Vina-family affinities for small fragments in a kinase pocket
    assert scores['score_min'].between(-15, 0).all()
    print(scores[['ligand', 'collection_key', 'score_min', 'score_average']].sort_values('score_min').to_string(index=False))

    # Every pose was written and converted to SDF
    for workunit_id, subjob_id in screen.subjobs():
        with tarfile.open(screen.job_dir / "qvina02" / "logs" / workunit_id / f"{subjob_id}.tar.gz") as tar:
            poses = [m.name for m in tar.getmembers() if m.name.endswith("output.sdf")]
        assert len(poses) == 3

    status = screen.run("afvs_get_status.py").stdout
    assert "  - Succeeded     : 6" in status
    assert "  - Failed        : 0" in status
    print(status)
