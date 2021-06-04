import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


RESULTS_DIR = pathlib.Path("save_results")

_min_traces_per_experiment = []
_experiments_that_did_not_converge = 0
for _dir in RESULTS_DIR.iterdir():
    _experiment_id = int(_dir.name)
    _ranks = np.load(_dir / "avg_rank_ASCAD_desync0.npy")
    _traces_with_rank_0 = np.where(_ranks <= 0)[0]
    if len(_traces_with_rank_0) > 0:
        _min_traces_per_experiment.append(
            _traces_with_rank_0[0]
        )
    else:
        _experiments_that_did_not_converge += 1

plt.hist(np.asarray(_min_traces_per_experiment), bins=100)
plt.show()

if _experiments_that_did_not_converge > 0:
    print(
        f"Note that {_experiments_that_did_not_converge} experiments did not converge"
    )



