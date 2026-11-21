# Experiment environments

The final pipeline only needs `requirements-final.txt` in the project root.

- `requirements-exp011.txt`: LightGBM, CatBoost and XGBoost for the boosted-tree comparison. Install from the project root with `python -m pip install --target .deps-exp011 --no-deps -r configs/environments/requirements-exp011.txt`.
- `requirements-exp015.txt`: the GNN experiment, with CUDA wheels. It needs its own environment.

`environment.yaml` in the root is the environment for the notebooks.
