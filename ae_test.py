#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 

data_path = "/data/qgong/data/AE/CICDDoS2019/Data/cicddos"

X_train = np.load(data_path+"/X_train.npy")
y_train = np.load(data_path+"/y_train.npy")
X_test = np.load(data_path+"/X_test.npy")
y_test = np.load(data_path+"/y_test.npy")

X_adv = np.load(data_path+"/Adv/Adv_X_test.npy")
y_adv = np.copy(y_test)

print(np.unique(y_train))
print(np.unique(y_test))
print(np.unique(y_adv))

X_train.shape, X_test.shape, y_train.shape, y_test.shape, X_adv.shape, y_adv.shape


# In[2]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, InputLayer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score


# Parameters
input_dim = 78  # Replace with the number of features in your tabular dataset
output_dim = 11  # Replace with the number of classes


from sklearn.metrics import precision_score, recall_score, f1_score

# Define a more practical classifier architecture
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


pretrained_classifier = create_classifier(input_dim, output_dim)
pretrained_classifier.load_weights('classifier.weights.h5')
# pretrained_classifier.compile()

pretrained_classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Verify the loaded model works
loss, accuracy = pretrained_classifier.evaluate(X_test, y_test, verbose=0)

# Predict class probabilities and get the predicted classes
y_pred_prob = pretrained_classifier.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)

# Compute precision, recall, and F1 score using scikit-learn
precision = precision_score(y_test, y_pred, average='weighted')  # Weighted for multi-class
recall = recall_score(y_test, y_pred, average='weighted')        # Weighted for multi-class
f1 = f1_score(y_test, y_pred, average='weighted')                # Weighted for multi-class

# Print the metrics
print(f"Accuracy: {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall: {recall:.6f}")
print(f"F1 Score: {f1:.6f}")


# In[ ]:


from alibi_detect.ad import AdversarialAE
from alibi_detect.utils.saving import save_detector

# Define encoder and decoder (same as before or custom)
encoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(input_dim,)),  # Replace `input_dim` with the number of features in your dataset
        Dense(512, activation='relu'),
        # Dropout(0.2),       
        Dense(256, activation='relu'),
        # Dropout(0.2),
        Dense(128, activation='relu'),
        # Dropout(0.2),
        Dense(64)  # Latent representation size
    ]
)

# Define custom MLP decoder
decoder_net = tf.keras.Sequential(
    [
        InputLayer(input_shape=(64,)),  # Match the latent representation size
        Dense(128, activation='relu'),
        # Dropout(0.2),
        Dense(256, activation='relu'),
        # Dropout(0.2),
        Dense(512, activation='relu'),
        # Dropout(0.2),
        Dense(input_dim)  # Match the original input dimension
    ]
)

# Initialize AdvAE with the pre-trained classifier
ad = AdversarialAE(
    encoder_net=encoder_net,
    decoder_net=decoder_net,
    model=pretrained_classifier,
    temperature=0.5
)

# Train the AdvAE
ad.fit(X_train, epochs=15, batch_size=64, verbose=True)

# Save the trained AdvAE detector
save_detector(ad, 'adv_ae_detector_with_pretrained_classifier')

