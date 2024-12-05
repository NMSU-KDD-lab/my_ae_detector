#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, InputLayer
from tensorflow.keras.models import Model
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from alibi_detect.ad import AdversarialAE
from alibi_detect.utils.saving import save_detector, load_detector
import argparse
import sys

# Command-line arguments
parser = argparse.ArgumentParser(description="Train and predict using an Adversarial Autoencoder.")
parser.add_argument("--data_path", type=str, default="/data/qgong/data/AE/CICDDoS2019/Data/cicddos", help="Path to the dataset.")
parser.add_argument("--model_save_path", type=str, default="adv_ae_detector_with_pretrained_classifier", help="Path to save the trained AdvAE detector.")
parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training advAE.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and prediction.")
parser.add_argument("--threshold_perc", type=float, default=90.0, help="Threshold percentile for adversarial detection.")
parser.add_argument("--mlp_model_path", type=str, default="classifier.weights.h5", help="Path to save or load the MLP model weights.")
parser.add_argument("--train", action="store_true", help="Run the training phase and save the model.")
parser.add_argument("--eval", action="store_true", help="Run the evaluation phase, including infer_threshold and predictions.")
parser.add_argument("--mlp", action="store_true", help="If provided, train the MLP model; otherwise, load the MLP model from the specified path.")
parser.add_argument("--mlp_epochs", type=int, default=20, help="Number of epochs for training MLP classifier.")

args = parser.parse_args()

print(args)

if not args.train and not args.eval:
    print("Error: You must specify at least one of --train or --eval.")
    sys.exit(1)

# Load data
data_path = args.data_path
X_train = np.load(f"{data_path}/X_train.npy")
y_train = np.load(f"{data_path}/y_train.npy")
X_test = np.load(f"{data_path}/X_test.npy")
y_test = np.load(f"{data_path}/y_test.npy")
X_adv = np.load(f"{data_path}/Adv/Adv_X_test.npy")
y_adv = np.copy(y_test)

assert X_train.shape[0] == y_train.shape[0], "Mismatch: X_train and y_train have different sample sizes."
assert X_test.shape[0] == y_test.shape[0], "Mismatch: X_test and y_test have different sample sizes."
assert X_adv.shape[0] == y_adv.shape[0], "Mismatch: X_adv and y_adv have different sample sizes."
assert X_train.shape[1] == X_test.shape[1] == X_adv.shape[1], "Mismatch: Features (columns) in X_train, X_test, and X_adv do not match."

# Data shapes and unique labels
print(f"Unique labels in training data: {np.unique(y_train)}")
print(f"Unique labels in test data: {np.unique(y_test)}")
print(f"Unique labels in adversarial data: {np.unique(y_adv)}")
print(f"Data shapes: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}, {X_adv.shape}, {y_adv.shape}")

# Model parameters
input_dim = X_train.shape[1]  # Features in dataset
output_dim = len(np.unique(y_train)) # Number of classes

# Define classifier architecture
def create_classifier(input_dim, output_dim):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(output_dim, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

classifier = create_classifier(input_dim, output_dim)



if args.mlp:
    print("Training MLP classifier...")
    classifier = create_classifier(input_dim, output_dim)
    classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

    # Train the classifier
    classifier.fit(X_train, y_train, epochs=args.mlp_epochs, batch_size=64, validation_split=0.2)

    # Save the classifier weights
    classifier.save_weights(args.mlp_model_path)
    print(f"Classifier training complete. Weights saved as {args.mlp_model_path}.")
else:
    # Load pretrained classifier
    classifier.load_weights(args.mlp_model_path)
    classifier.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Classifier and Weights are loded from '{args.mlp_model_path}.")


# Training phase
if args.train:
    print("Starting training phase...")
    # Define AdvAE architecture
    encoder_net = tf.keras.Sequential([
        InputLayer(input_shape=(input_dim,)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64)
    ])

    decoder_net = tf.keras.Sequential([
        InputLayer(input_shape=(64,)),
        Dense(128, activation='relu'),
        Dense(256, activation='relu'),
        Dense(512, activation='relu'),
        Dense(input_dim)
    ])

    # Initialize AdvAE detector
    adv_ae = AdversarialAE(
        encoder_net=encoder_net,
        decoder_net=decoder_net,
        model=classifier,
        temperature=0.5
    )

    # Train AdvAE
    adv_ae.fit(X_train, epochs=args.epochs, batch_size=args.batch_size, verbose=True)

    # Save AdvAE detector
    save_detector(adv_ae, args.model_save_path)
    print(f"Model saved at {args.model_save_path}")

# Evaluation phase
if args.eval:
    print("Starting evaluation phase...")

    # Load AdvAE detector
    adv_ae = load_detector(args.model_save_path)

    # Infer threshold
    adv_ae.infer_threshold(X_train, threshold_perc=args.threshold_perc, batch_size=args.batch_size)

    # Predict adversarial instances
    predictions = adv_ae.predict(X_adv, batch_size=args.batch_size, return_instance_score=True)

    # Print prediction results
    print("Instance Scores (first 20):", predictions['data']['instance_score'][:20])
    print("Is Adversarial (first 20):", predictions['data']['is_adversarial'][:20])
    print(f"Prediction data shapes: {predictions['data']['instance_score'].shape}, {predictions['data']['is_adversarial'].shape}")

    # Evaluate metrics
    y_true = np.ones_like(predictions['data']['is_adversarial'])  # All examples are adversarial
    y_pred = predictions['data']['is_adversarial']  # Predicted labels

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    # Print metrics
    print(f"Evaluation Metrics:")
    print(f"  - Precision: {precision:.6f}")
    print(f"  - Recall:    {recall:.6f}")
    print(f"  - F1 Score:  {f1:.6f}")
    print(f"  - Accuracy:  {accuracy:.6f}")
    if len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, predictions['data']['instance_score'])
        print(f"  - AUC-ROC:   {auc_roc:.6f}")
    else:
        print("  - AUC-ROC:   Not defined (only one class present in y_true).")

# If neither --train nor --eval is provided
if not args.train and not args.eval and not args.mlp:
    print("Please provide --train, --eval, --mlp, or all flags to execute the respective phases.")
