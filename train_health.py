import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

dataset = pd.read_csv("data/insurance.csv")
print("shape:", dataset.shape)
dataset = pd.get_dummies(dataset, columns=["sex", "smoker", "region"], drop_first=True)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop("expenses")
test_labels = test_dataset.pop("expenses")

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))

model = keras.Sequential([
    normalizer,
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mae", metrics=["mae"])

history = model.fit(
    train_dataset, train_labels,
    epochs=120, batch_size=32,
    validation_split=0.2, verbose=0,
)

loss, mae = model.evaluate(test_dataset, test_labels, verbose=2)
print("MAE:", mae)
print("PASS" if mae < 3500 else "FAIL")
