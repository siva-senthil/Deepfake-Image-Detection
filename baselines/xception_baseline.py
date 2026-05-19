"""
Xception Baseline — Deep Fake Detection
Dataset: Deep Fake Face Detection (Kaggle)
Paper: https://doi.org/10.1007/s10791-025-09586-2
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import (BinaryAccuracy, Precision, Recall,
                                      FalseNegatives, FalsePositives,
                                      TrueNegatives, TruePositives, AUC)
from tensorflow.math import confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

# ── Dataset paths (update these) ──────────────────────────────────────────────
TRAINING_DIR   = "dataset/Train"
VALIDATION_DIR = "dataset/Validation"
TEST_DIR       = "dataset/Test"

IMG_SIZE  = 224
IMG_BATCH = 32

# ── Data generators ───────────────────────────────────────────────────────────
training_datagen   = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen       = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True, class_mode='binary', batch_size=IMG_BATCH
)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True, class_mode='binary', batch_size=IMG_BATCH
)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    class_mode='binary', batch_size=IMG_BATCH, shuffle=False
)

# ── Model ─────────────────────────────────────────────────────────────────────
base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_model.layers:
    layer.trainable = False   # freeze pretrained weights

x = Flatten(name='flatten')(base_model.output)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
out = Dense(1, activation='sigmoid', name='output')(x)

model = Model(base_model.input, out)
model.summary()
model.compile(
    loss=binary_crossentropy,
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath='model_xception.keras', monitor='val_loss',
                    mode='min', save_best_only=True, verbose=1)
]

# ── Training ──────────────────────────────────────────────────────────────────
history = model.fit(
    train_generator, epochs=100,
    validation_data=validation_generator,
    verbose=1, callbacks=callbacks
)

# ── Plot learning curves ──────────────────────────────────────────────────────
plt.figure(figsize=(25, 8))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy'); plt.xlabel('Epoch'); plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()
plt.savefig('xception_learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_model(model_path, test_generator, show_plots=False):
    loaded_model = tf.keras.models.load_model(model_path)
    true_labels = test_generator.labels
    predictions_sigmoid = loaded_model.predict(test_generator, verbose=1)

    sigmoid_threshold = np.arange(0.2, 0.7, 0.001)
    fp_fn = []
    for t in sigmoid_threshold:
        preds = tf.where(predictions_sigmoid > t, 1, 0)
        cm = confusion_matrix(true_labels, preds)
        fp_fn.append(cm[0][1] + cm[1][0])

    min_threshold = sigmoid_threshold[np.argmin(fp_fn)]
    print(f'Optimal threshold: {min_threshold:.3f}')

    bin_acc = BinaryAccuracy(threshold=min_threshold)
    bin_acc.update_state(true_labels, predictions_sigmoid)
    print(f'Binary Accuracy: {bin_acc.result().numpy():.3f}')

    precision = Precision(thresholds=min_threshold)
    recall    = Recall(thresholds=min_threshold)
    precision.update_state(true_labels, predictions_sigmoid)
    recall.update_state(true_labels, predictions_sigmoid)
    p, r = precision.result().numpy(), recall.result().numpy()
    f1 = 2 * (p * r) / (p + r)
    print(f'Precision: {p:.3f} | Recall: {r:.3f} | F1: {f1:.3f}')

    if show_plots:
        cm = confusion_matrix(true_labels, tf.where(predictions_sigmoid > min_threshold, 1, 0))
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm.numpy(), annot=True, fmt='d')
        plt.title(f'Confusion Matrix @ {min_threshold:.3f}')
        plt.ylabel('True'); plt.xlabel('Predicted')
        plt.savefig('xception_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()

    return min_threshold


evaluate_model('model_xception.keras', test_generator, show_plots=True)
