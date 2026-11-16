# Optional experiment environments

The verified final CPU pipeline uses `requirements-final.txt` at the project root.

- `requirements-exp011.txt`: optional LightGBM/CatBoost/XGBoost experiment packages. From the project root: `python -m pip install --target .deps-exp011 --no-deps -r configs/environments/requirements-exp011.txt`. Its included requirements file is resolved relative to this folder.
- `requirements-exp015.txt`: archived relational GNN environment pins, including CUDA-specific wheels. Use its dedicated environment and documented indexes; these packages are not required for final H2/P021 inference.

`environment.yaml` at the root remains the general notebook/EDA environment.
