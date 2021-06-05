import pathlib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import colors
import plotly.express as px
from matplotlib.ticker import PercentFormatter


RESULTS_DIR = pathlib.Path("save_results_original")

ranks_fig_non_converged = go.Figure()
ranks_fig_converged = go.Figure()
_min_traces_per_experiment = []
for _experiment_id in range(100):
    _dir = RESULTS_DIR / str(_experiment_id)
    # noinspection PyTypeChecker
    _ranks = np.load(_dir / "avg_rank_ASCAD_desync0.npy")
    _traces_with_rank_0 = np.where(_ranks <= 0)[0]
    if len(_traces_with_rank_0) > 0:
        _min_traces_per_experiment.append(
            _traces_with_rank_0[0]
        )
        ranks_fig_converged.add_trace(
            go.Scatter(
                x=np.arange(len(_ranks)),
                y=_ranks,
                mode='lines',
                name=f"exp_{_experiment_id:03d}"
            )
        )
    else:
        _min_traces_per_experiment.append(
            np.inf
        )
        ranks_fig_non_converged.add_trace(
            go.Scatter(
                x=np.arange(len(_ranks)),
                y=_ranks,
                mode='lines',
                name=f"exp_{_experiment_id:03d}"
            )
        )

ranks_fig_non_converged.show()
ranks_fig_converged.show()

_experiments_that_did_not_converge = [i for i, v in enumerate(_min_traces_per_experiment) if v == np.inf]
_min_traces_for_converged_experiments = [v for i, v in enumerate(_min_traces_per_experiment) if v != np.inf]
plt.hist(np.asarray(_min_traces_for_converged_experiments), bins=100)
plt.show()
if bool(_experiments_that_did_not_converge):
    print(
        f"Note that {len(_experiments_that_did_not_converge)} experiments did not converge"
    )
    print(_experiments_that_did_not_converge)


_sorted_ids = np.argsort(_min_traces_per_experiment)

for _id in _sorted_ids:
    print("experiment", f"{_id:03d}", ":: min traces", _min_traces_per_experiment[_id])



