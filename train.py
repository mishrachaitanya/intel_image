import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import numpy as np

# Paths
TRAIN_DIR = 'data/train'
VAL_DIR = 'data/validation'
MODEL_PATH = os.path.join(os.getcwd(), 'model.weights.h5')

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 5
IMG_SIZE = (128, 128)
LR = 0.001

# Disable GPU (optional, since you're on CPU anyway)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Data loading and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Load a pretrained model
base_model = tf.keras.applications.ResNet50(
    weights='imagenet', include_top=False, input_shape=(128, 128, 3)
)
base_model.trainable = False  # Freeze base

# Add classification head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(2, activation='softmax')  # 2 classes
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Evaluation
val_generator.reset()
preds = model.predict(val_generator, verbose=0)
pred_labels = np.argmax(preds, axis=1)
true_labels = val_generator.classes

acc = accuracy_score(true_labels, pred_labels)
print(f"\nFinal Validation Accuracy: {acc:.4f}")

# Save weights
model.save_weights(MODEL_PATH)
print(f"\n Model saved as: {MODEL_PATH}")
