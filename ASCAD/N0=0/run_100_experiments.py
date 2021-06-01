import tensorflow
import pathlib
import shutil
import os
import gc
import numpy as np
from IPython.display import SVG
print(tensorflow.__version__)

SAVE_DIR = pathlib.Path("save_results")
SAVE_DIR.mkdir(exist_ok=True, parents=True)
RESULT_DIR = pathlib.Path(".")

for i in range(100):
    gc.collect()

    _save_dir = SAVE_DIR / str(i)
    if _save_dir.exists():
        print(f"Skipping experiment {i}")
        continue
    _save_dir.mkdir()

    print(f"Performing experiment {i}")

    print("... Cleaning")
    _model_file = RESULT_DIR / "ASCAD_trained_models/ASCAD_desync0"
    _fig_file = RESULT_DIR / "fig/rankASCAD_desync0_300trs_100att.svg"
    _predictions_file = RESULT_DIR / "model_predictions/predictions_ASCAD_desync0.npy"
    _avg_rank_file = RESULT_DIR / "model_predictions/avg_rank_ASCAD_desync0.npy"
    _history_file = RESULT_DIR / "training_history/history_ASCAD_desync0"
    if _model_file.exists():
        _model_file.unlink()
    if _fig_file.exists():
        _fig_file.unlink()
    if _predictions_file.exists():
        _predictions_file.unlink()
    if _avg_rank_file.exists():
        _avg_rank_file.unlink()
    if _history_file.exists():
        _history_file.unlink()

    print("... Running code")
    os.system("python cnn_architecture.py")

    print("... Save results")
    shutil.copy(_model_file, _save_dir / _model_file.name)
    shutil.copy(_fig_file, _save_dir / _fig_file.name)
    shutil.copy(_predictions_file, _save_dir / _predictions_file.name)
    shutil.copy(_avg_rank_file, _save_dir / _avg_rank_file.name)
    shutil.copy(_history_file, _save_dir / _history_file.name)


