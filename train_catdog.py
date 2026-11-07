import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "8"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["MALLOC_ARENA_MAX"] = "2"

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

tf.random.set_seed(42)
np.random.seed(42)

PATH = "data/cats_and_dogs"
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")
test_dir = os.path.join(PATH, "test")

total_train = sum(len(files) for _, _, files in os.walk(train_dir))
total_val = sum(len(files) for _, _, files in os.walk(validation_dir))

batch_size = 32
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)
validation_image_generator = ImageDataGenerator(rescale=1.0 / 255)

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

print("total train", total_train, "val", total_val)
print("train samples", train_data_gen.samples, "val samples", val_data_gen.samples)

model = Sequential([
    Conv2D(16, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=[
        ModelCheckpoint("catdog_best.keras", monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=4,
                      restore_best_weights=True, verbose=1),
    ],
    verbose=2,
)

model.save("catdog_final.keras")
with open("catdog_history.json", "w") as f:
    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)

probabilities = model.predict(test_data_gen, verbose=0).flatten()
answers = [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
           1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
           1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
           1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
           0, 0, 0, 0, 0, 0]
correct = sum(round(p) == a for p, a in zip(probabilities, answers))
percentage = correct / len(answers) * 100
print(f"TEST ACCURACY: {percentage:.2f}%  ({correct}/{len(answers)})")
print("PASS" if percentage >= 63 else "FAIL")
np.save("catdog_test_probs.npy", probabilities)
