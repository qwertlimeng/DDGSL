# DDGSL: Direction-Aware Dual-Graph Structure Learning for Multivariate Time Series Anomaly Detection

This repository provides the PyTorch implementation of DDGSL for multivariate time series anomaly detection (MTSAD). DDGSL combines direction-aware dynamic graph learning, cosine-similarity graph construction, adaptive graph fusion, and recursive spatiotemporal interaction blocks (STIBs) to model intervariable dependencies and temporal patterns.

## 1. Requirements

Python 3.10 and a CUDA-enabled GPU are recommended. Install a compatible PyTorch version following the [official instructions](https://pytorch.org/get-started/previous-versions/), followed by:

```bash
pip install -r requirements.txt
```

Training and evaluation require CUDA.

## 2. Datasets

The implementation supports multivariate time series supplied in a common CSV format. The datasets used in the paper can be obtained from:

- MSL:[GDN repository](https://github.com/d-ailin/GDN) (partial demonstration dataset).
- SWaT and WADI:[SUTD iTrust](https://www.sutd.edu.sg/itrust/itrust-labs/datasets/).
- HAI: [Official HAI repository](https://github.com/icsdataset/hai).

Prepare the following files under `data/`:

```text
data/
├── train.csv
├── test.csv
└── test_label.csv  # Optional
```

The training data must contain normal observations. Training and test files must have identical ordered feature columns. Binary test labels (0: normal, 1: anomaly) may be provided separately or included in `test.csv`.

Configure data paths, feature columns, optional timestamps and sequence identifiers, and model parameters in `configs/default.yaml`.

The optional `preprocess.py` utility prepares existing labeled numeric CSV files. It supports explicit missing-value handling and downsampling.

## 3. Training and Evaluation

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Train DDGSL:

```bash
python train.py --config configs/default.yaml
```

Evaluate the trained model:

```bash
python evaluate.py --checkpoint runs/RUN_ID/selected_model.pt --config configs/default.yaml
```

Replace `RUN_ID` with the generated training directory.

The evaluation reports Precision, Recall, F1, and AUROC when test labels are available. Without labels, anomaly scores and binary predictions are saved.

## 4. Experimental Settings

The implementation follows these procedures:

- Data splitting: Normal training sequences are chronologically divided into 70% training and 30% validation before window construction.
- Normalization: Min-max parameters are estimated exclusively from training observations, unless previously normalized inputs are explicitly declared.
- Optimization: The objective combines prediction MSE, graph temporal smoothness, and directional regularization.
- Model selection: The checkpoint is selected using normal-validation MSE.
- Anomaly detection: Prediction residuals are standardized using normal-validation median and MAD. The maximum variable score is compared against a validation-based quantile threshold (\(\rho=0.995\) by default).
- Evaluation: Pointwise metrics are computed without point adjustment, POT, or temporal postprocessing.

The random seed and model hyperparameters are specified in `configs/default.yaml`. 


