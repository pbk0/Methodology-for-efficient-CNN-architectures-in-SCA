import pathlib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pickle
from matplotlib import colors
import plotly.express as px
from matplotlib.ticker import PercentFormatter


RESULTS_DIR = pathlib.Path("save_results_with_early_stopping")

ranks_fig_non_converged = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text="Average Rank (non-converged)")
    )
)
ranks_fig_converged = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text="Average Rank (converged)")
    )
)
train_loss_fig_non_converged = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text="Train Loss (non-converged)")
    )
)
train_loss_fig_converged = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text="Train Loss (converged)")
    )
)
val_loss_fig_non_converged = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text="Validation Loss (non-converged)")
    )
)
val_loss_fig_converged = go.Figure(
    layout=go.Layout(
        title=go.layout.Title(text="Validation Loss (converged)")
    )
)
_min_traces_per_experiment = []
for _experiment_id in range(100):
    _dir = RESULTS_DIR / str(_experiment_id)
    # noinspection PyTypeChecker
    _ranks = np.load(_dir / "avg_rank_ASCAD_desync0.npy")
    with open((_dir / 'history_ASCAD_desync0').as_posix(), 'rb') as file_pi:
        print(_dir / 'history_ASCAD_desync0')
        _history = pickle.load(file_pi)
    _train_loss = _history['loss']
    _val_loss = _history['val_loss']
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
        train_loss_fig_converged.add_trace(
            go.Scatter(
                x=np.arange(len(_train_loss)),
                y=_train_loss,
                mode='lines',
                name=f"exp_{_experiment_id:03d}"
            )
        )
        val_loss_fig_converged.add_trace(
            go.Scatter(
                x=np.arange(len(_val_loss)),
                y=_val_loss,
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
        train_loss_fig_non_converged.add_trace(
            go.Scatter(
                x=np.arange(len(_train_loss)),
                y=_train_loss,
                mode='lines',
                name=f"exp_{_experiment_id:03d}"
            )
        )
        val_loss_fig_non_converged.add_trace(
            go.Scatter(
                x=np.arange(len(_val_loss)),
                y=_val_loss,
                mode='lines',
                name=f"exp_{_experiment_id:03d}"
            )
        )

ranks_fig_non_converged.show()
ranks_fig_converged.show()
train_loss_fig_non_converged.show()
train_loss_fig_converged.show()
val_loss_fig_non_converged.show()
val_loss_fig_converged.show()

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



