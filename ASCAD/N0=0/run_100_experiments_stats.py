import pathlib
import numpy as np

RESULTS_DIR = pathlib.Path("save_results")

for _dir in RESULTS_DIR.iterdir():
    _experiment_id = int(_dir.name)
    _ranks = np.load(_dir / "avg_rank_ASCAD_desync0.npy")
    print(np.where(_ranks<=0))