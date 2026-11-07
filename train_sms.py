import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

train_data = pd.read_csv("data/train-data.tsv", sep="\t", names=["label", "message"])
valid_data = pd.read_csv("data/valid-data.tsv", sep="\t", names=["label", "message"])
print(train_data.shape, valid_data.shape, train_data["label"].value_counts().to_dict())

train_text = train_data["message"].values
valid_text = valid_data["message"].values
train_labels = (train_data["label"].values == "spam").astype("float32")
valid_labels = (valid_data["label"].values == "spam").astype("float32")

max_features = 10000
sequence_length = 120

vectorizer = layers.TextVectorization(
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)
vectorizer.adapt(train_text)

model = keras.Sequential([
    vectorizer,
    layers.Embedding(max_features, 32),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()
history = model.fit(
    train_text, train_labels,
    epochs=12, batch_size=64,
    validation_data=(valid_text, valid_labels),
    verbose=2,
)

loss, acc = model.evaluate(valid_text, valid_labels, verbose=0)
print("val acc", acc)


def predict_message(pred_text):
    batch = tf.constant([pred_text])
    probability = float(model.predict(batch, verbose=0)[0][0])
    return [probability, "spam" if probability > 0.5 else "ham"]


test_messages = ["how are you doing today",
                 "sale today! to stop texts call 98912460324",
                 "i dont want to go. can we try it a different day? available sat",
                 "our new mobile video service is live. just install on your phone to start watching.",
                 "you have won £1000 cash! call to claim your prize.",
                 "i'll bring it tomorrow. don't forget the milk.",
                 "wow, is your arm alright. that happened to me one time too"]
test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
for m, a in zip(test_messages, test_answers):
    p = predict_message(m)
    print(f"{p[1]:5s} (true {a:5s}) {p[0]:.4f} :: {m}")
