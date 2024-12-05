# Adversarial Autoencoder (AdvAE) and MLP Training Script

This script provides functionality to train and evaluate a Multi-Layer Perceptron (MLP) and an Adversarial Autoencoder (AdvAE) for adversarial instance detection. It is highly configurable via command-line arguments for flexible usage.

---

## Features
- Train an MLP classifier.
- Load pre-trained MLP weights.
- Train an AdvAE detector.
- Evaluate the AdvAE detector on adversarial data.

---

## Requirements
- Python 3.7+
- Required Python libraries:
  - `numpy`
  - `tensorflow`
  - `sklearn`
  - `alibi-detect`

Install dependencies:
```
conda create -n my_ae_detector python=3.9
conda activate my_ae_detector
pip install alibi-detect[tensorflow]
pip install tensorflow[and-cuda]==2.14.1
```

### 2. Example Commands

#### Train MLP and AdvAE
```bash
python advAE_script.py --mlp --train --data_path "/data/qgong/data/AE/CICDDoS2019/Data/cicddos" --mlp_model_path "mlp_classifier.weights.h5" --model_save_path "adv_ae_detector_with_pretrained_classifier"
```

#### Load Pre-trained MLP and Train AdvAE
```bash
python advAE_script.py --train --data_path "/data/qgong/data/AE/CICDDoS2019/Data/cicddos" --mlp_model_path "mlp_classifier.weights.h5" --model_save_path "adv_ae_detector_with_pretrained_classifier"
```

#### Train MLP, Train AdvAE, and Perform Prediction
```bash
python advAE_script.py --mlp --train --eval --data_path "/data/qgong/data/AE/CICDDoS2019/Data/cicddos" --mlp_model_path "mlp_classifier.weights.h5" --model_save_path "adv_ae_detector_with_pretrained_classifier" --threshold_perc 90.0
```

#### Load Pre-trained MLP and AdvAE, and Perform Prediction
```bash
python advAE_script.py --eval --data_path "/data/qgong/data/AE/CICDDoS2019/Data/cicddos" --mlp_model_path "mlp_classifier.weights.h5" --model_save_path "adv_ae_detector_with_pretrained_classifier" --threshold_perc 90.0
```

---

## Data Requirements
The dataset should be structured as follows:
- `X_train.npy`: Features for training.
- `y_train.npy`: Labels for training.
- `X_test.npy`: Features for testing.
- `y_test.npy`: Labels for testing.
- `Adv/Adv_X_test.npy`: Features for adversarial examples.

Ensure all `.npy` files are present in the directory specified by the `--data_path` argument.

---

## Workflow

### Step 1: Train MLP
If `--mlp` is provided:
1. Train the MLP classifier using the training dataset (`X_train`, `y_train`).
2. Save the trained weights to the file specified by `--mlp_model_path`.

### Step 2: Train AdvAE
If `--train` is provided:
1. Use the MLP classifier (trained or loaded) as a pre-trained model.
2. Train the AdvAE detector using the training dataset.
3. Save the trained AdvAE model to the path specified by `--model_save_path`.

### Step 3: Evaluate AdvAE
If `--eval` is provided:
1. Load the trained AdvAE model from the path specified by `--model_save_path`.
2. Infer the detection threshold using the training dataset.
3. Predict adversarial instances on the adversarial dataset (`Adv/Adv_X_test.npy`).

---

## Outputs
1. **MLP Model Weights**:
   Saved at the path specified by `--mlp_model_path` (default: `classifier.weights.h5`).

2. **AdvAE Detector**:
   Saved at the path specified by `--model_save_path` (default: `adv_ae_detector_with_pretrained_classifier`).

3. **Evaluation Metrics**:
   - Instance scores for adversarial detection.
   - Adversarial prediction results.
---
