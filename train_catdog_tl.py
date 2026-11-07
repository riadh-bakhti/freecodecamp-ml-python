import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["MALLOC_ARENA_MAX"] = "2"

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

tf.random.set_seed(42)
np.random.seed(42)

PATH = "data/cats_and_dogs"
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")
test_dir = os.path.join(PATH, "test")

total_train = sum(len(files) for _, _, files in os.walk(train_dir))
total_val = sum(len(files) for _, _, files in os.walk(validation_dir))

batch_size = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

train_image_generator = ImageDataGenerator(
    preprocessing_function=preprocess,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)
validation_image_generator = ImageDataGenerator(preprocessing_function=preprocess)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size, directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode="binary", seed=42)
val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size, directory=validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode="binary", seed=42)
test_data_gen = validation_image_generator.flow_from_directory(
    batch_size=1, directory=os.path.dirname(test_dir),
    classes=[os.path.basename(test_dir)],
    target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode=None, shuffle=False)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False,
    weights="imagenet")
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

print("=== Stage 1: training the classifier head ===")
h1 = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=8,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=[
        ModelCheckpoint("catdog_tl.keras", monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=3,
                      restore_best_weights=True, verbose=1),
    ],
    verbose=2,
)

print("=== Stage 2: fine-tuning the top of MobileNetV2 ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy", metrics=["accuracy"])
h2 = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=8,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=[
        ModelCheckpoint("catdog_tl.keras", monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=3,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                          patience=1, min_lr=1e-7, verbose=1),
    ],
    verbose=2,
)

model.save("catdog_final_tl.keras")
with open("catdog_history_tl.json", "w") as f:
    json.dump({k: [float(x) for x in v] for k, v in h2.history.items()}, f)

# --- evaluation with horizontal-flip test-time augmentation ---
test_data_gen.reset()
images = []
for i in range(len(test_data_gen)):
    images.append(test_data_gen[i][0])
images = np.vstack(images)
flipped = np.ascontiguousarray(images[:, :, ::-1, :])
probabilities = (model.predict(images, verbose=0).flatten()
                 + model.predict(flipped, verbose=0).flatten()) / 2.0

answers = [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
           1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
           1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
           1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
           0, 0, 0, 0, 0, 0]
correct = sum(round(p) == a for p, a in zip(probabilities, answers))
percentage = correct / len(answers) * 100
print(f"TEST ACCURACY (TTA): {percentage:.2f}%  ({correct}/{len(answers)})")
print("PASS" if percentage >= 63 else "FAIL")
np.save("catdog_test_probs_tl.npy", probabilities)
