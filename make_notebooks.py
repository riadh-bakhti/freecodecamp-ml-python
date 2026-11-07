"""Build the five freeCodeCamp solution notebooks."""
import json
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def nb(cells):
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = cells
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": []},
    }
    return notebook


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ---------------------------------------------------------------- RPS
rps_game = open(os.path.join(HERE, "RPS_game.py")).read()
rps_player = open(os.path.join(HERE, "RPS.py")).read()

rps = nb([
    md("# Rock Paper Scissors\n\n"
       "Solution for the freeCodeCamp *Machine Learning with Python* project.\n\n"
       "The player predicts the opponent's next move with an ensemble of models "
       "(Markov chains of order 1-5 on the opponent's history, plus models of the "
       "three adaptive bots) and plays the move that beats the prediction."),
    code("import random\n\n" + rps_game),
    md("## Player"),
    code(rps_player),
    md("## Test against the four bots"),
    code(
        "for name, bot in [('quincy', quincy), ('abbey', abbey),\n"
        "                  ('kris', kris), ('mrugesh', mrugesh)]:\n"
        "    rate = play(player, bot, 1000)\n"
        "    print(f'{name}: {rate:.1f}% - ' + ('PASS' if rate >= 60 else 'FAIL'))\n"
    ),
])

# ---------------------------------------------------------------- Cat / Dog
catdog = nb([
    md("# Cat and Dog Image Classifier\n\n"
       "CNN trained on the freeCodeCamp cats-and-dogs dataset. "
       "Target: at least 63% of the 50 test images correctly classified."),
    code(
        "try:\n"
        "    %tensorflow_version 2.x\n"
        "except Exception:\n"
        "    pass\n"
        "import tensorflow as tf\n"
        "from tensorflow.keras.models import Sequential\n"
        "from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D\n"
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n"
        "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\n"
        "import os\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "tf.random.set_seed(42)\n"
        "np.random.seed(42)"
    ),
    code(
        "# Get project files (skipped when the data is already present locally).\n"
        "if not os.path.isdir('data/cats_and_dogs'):\n"
        "    if not os.path.exists('cats_and_dogs'):\n"
        "        !wget -q https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip\n"
        "        !unzip -q -o cats_and_dogs.zip\n"
        "    PATH = 'cats_and_dogs'\n"
        "else:\n"
        "    PATH = 'data/cats_and_dogs'\n"
        "\n"
        "train_dir = os.path.join(PATH, 'train')\n"
        "validation_dir = os.path.join(PATH, 'validation')\n"
        "test_dir = os.path.join(PATH, 'test')\n"
        "\n"
        "total_train = sum([len(files) for r, d, files in os.walk(train_dir)])\n"
        "total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])\n"
        "total_test = len([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])\n"
        "\n"
        "batch_size = 32\n"
        "epochs = 15\n"
        "IMG_HEIGHT = 150\n"
        "IMG_WIDTH = 150\n"
        "print(total_train, total_val, total_test)"
    ),
    code(
        "# 3\n"
        "train_image_generator = ImageDataGenerator(rescale=1.0 / 255)\n"
        "validation_image_generator = ImageDataGenerator(rescale=1.0 / 255)\n"
        "test_image_generator = ImageDataGenerator(rescale=1.0 / 255)\n"
        "\n"
        "train_data_gen = train_image_generator.flow_from_directory(\n"
        "    batch_size=batch_size, directory=train_dir,\n"
        "    target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary', seed=42)\n"
        "\n"
        "val_data_gen = validation_image_generator.flow_from_directory(\n"
        "    batch_size=batch_size, directory=validation_dir,\n"
        "    target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary', seed=42)\n"
        "\n"
        "test_data_gen = test_image_generator.flow_from_directory(\n"
        "    batch_size=1, directory=PATH, classes=['test'],\n"
        "    target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode=None, shuffle=False)"
    ),
    code(
        "# 4\n"
        "def plotImages(images_arr, probabilities=False):\n"
        "    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5, len(images_arr) * 3))\n"
        "    if probabilities is False:\n"
        "        for img, ax in zip(images_arr, axes):\n"
        "            ax.imshow(img)\n"
        "            ax.axis('off')\n"
        "    else:\n"
        "        for img, probability, ax in zip(images_arr, probabilities, axes):\n"
        "            ax.imshow(img)\n"
        "            ax.axis('off')\n"
        "            if probability > 0.5:\n"
        "                ax.set_title('%.2f' % (probability * 100) + '% dog')\n"
        "            else:\n"
        "                ax.set_title('%.2f' % ((1 - probability) * 100) + '% cat')\n"
        "    plt.show()\n"
        "\n"
        "sample_training_images, _ = next(train_data_gen)\n"
        "plotImages(sample_training_images[:5])"
    ),
    code(
        "# 5\n"
        "train_image_generator = ImageDataGenerator(\n"
        "    rescale=1.0 / 255,\n"
        "    rotation_range=20,\n"
        "    width_shift_range=0.15,\n"
        "    height_shift_range=0.15,\n"
        "    shear_range=0.1,\n"
        "    zoom_range=0.15,\n"
        "    horizontal_flip=True,\n"
        "    fill_mode='nearest')"
    ),
    code(
        "# 6\n"
        "train_data_gen = train_image_generator.flow_from_directory(\n"
        "    batch_size=batch_size, directory=train_dir,\n"
        "    target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary', seed=42)\n"
        "\n"
        "augmented_images = [train_data_gen[0][0][0] for i in range(5)]\n"
        "plotImages(augmented_images)"
    ),
    code(
        "# 7\n"
        "model = Sequential([\n"
        "    Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),\n"
        "    MaxPooling2D(2, 2),\n"
        "    Conv2D(32, (3, 3), activation='relu'),\n"
        "    MaxPooling2D(2, 2),\n"
        "    Conv2D(64, (3, 3), activation='relu'),\n"
        "    MaxPooling2D(2, 2),\n"
        "    Flatten(),\n"
        "    Dense(128, activation='relu'),\n"
        "    Dropout(0.5),\n"
        "    Dense(1, activation='sigmoid'),\n"
        "])\n"
        "model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),\n"
        "              loss='binary_crossentropy', metrics=['accuracy'])\n"
        "model.summary()"
    ),
    code(
        "# 8\n"
        "history = model.fit(\n"
        "    train_data_gen,\n"
        "    steps_per_epoch=total_train // batch_size,\n"
        "    epochs=epochs,\n"
        "    validation_data=val_data_gen,\n"
        "    validation_steps=total_val // batch_size,\n"
        "    callbacks=[EarlyStopping(monitor='val_accuracy', patience=4,\n"
        "                             restore_best_weights=True, verbose=1)],\n"
        "    verbose=2)"
    ),
    code(
        "# 9\n"
        "acc = history.history['accuracy']\n"
        "val_acc = history.history['val_accuracy']\n"
        "loss = history.history['loss']\n"
        "val_loss = history.history['val_loss']\n"
        "epochs_range = range(len(acc))\n"
        "\n"
        "plt.figure(figsize=(8, 8))\n"
        "plt.subplot(1, 2, 1)\n"
        "plt.plot(epochs_range, acc, label='Training Accuracy')\n"
        "plt.plot(epochs_range, val_acc, label='Validation Accuracy')\n"
        "plt.legend(loc='lower right')\n"
        "plt.title('Training and Validation Accuracy')\n"
        "\n"
        "plt.subplot(1, 2, 2)\n"
        "plt.plot(epochs_range, loss, label='Training Loss')\n"
        "plt.plot(epochs_range, val_loss, label='Validation Loss')\n"
        "plt.legend(loc='upper right')\n"
        "plt.title('Training and Validation Loss')\n"
        "plt.show()"
    ),
    code(
        "# 10 - predictions on the 50 test images\n"
        "probabilities = model.predict(test_data_gen).flatten()\n"
        "print(probabilities.shape)"
    ),
    code(
        "# 11 - DO NOT MODIFY CONTENTS\n"
        "answers = [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,\n"
        "           1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,\n"
        "           1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,\n"
        "           1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,\n"
        "           0, 0, 0, 0, 0, 0]\n"
        "\n"
        "correct = 0\n"
        "for probability, answer in zip(probabilities, answers):\n"
        "    if round(probability) == answer:\n"
        "        correct += 1\n"
        "\n"
        "percentage_identified = (correct / len(answers)) * 100\n"
        "passed_challenge = percentage_identified >= 63\n"
        "\n"
        "print(f'Your model correctly identified {round(percentage_identified, 2)}% of the images of cats and dogs.')\n"
        "if passed_challenge:\n"
        "    print('You passed the challenge!')\n"
        "else:\n"
        "    print(\"You haven't passed yet. Your model should identify at least 63% of the images. Keep trying. You will get it!\")"
    ),
])

# ---------------------------------------------------------------- Book KNN
book = nb([
    md("# Book Recommendation Engine using KNN"),
    code(
        "# import libraries (you may add additional imports but you may not have to)\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from scipy.sparse import csr_matrix\n"
        "from sklearn.neighbors import NearestNeighbors\n"
        "import matplotlib.pyplot as plt"
    ),
    code(
        "import os\n"
        "if os.path.exists('data/BX-Books.csv'):\n"
        "    books_filename = 'data/BX-Books.csv'\n"
        "    ratings_filename = 'data/BX-Book-Ratings.csv'\n"
        "else:\n"
        "    if not os.path.exists('BX-Books.csv'):\n"
        "        !wget -q https://cdn.freecodecamp.org/project-data/books/book-crossings.zip\n"
        "        !unzip -q -o book-crossings.zip\n"
        "    books_filename = 'BX-Books.csv'\n"
        "    ratings_filename = 'BX-Book-Ratings.csv'"
    ),
    code(
        "# import csv data into dataframes\n"
        "df_books = pd.read_csv(\n"
        "    books_filename,\n"
        "    encoding='ISO-8859-1',\n"
        "    sep=';',\n"
        "    header=0,\n"
        "    names=['isbn', 'title', 'author'],\n"
        "    usecols=['isbn', 'title', 'author'],\n"
        "    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})\n"
        "\n"
        "df_ratings = pd.read_csv(\n"
        "    ratings_filename,\n"
        "    encoding='ISO-8859-1',\n"
        "    sep=';',\n"
        "    header=0,\n"
        "    names=['user', 'isbn', 'rating'],\n"
        "    usecols=['user', 'isbn', 'rating'],\n"
        "    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})"
    ),
    md("## Filtering, pivot table and KNN model"),
    code(
        "# Remove users with fewer than 200 ratings and books with fewer than\n"
        "# 100 ratings (counts taken from the original dataset).\n"
        "user_counts = df_ratings['user'].value_counts()\n"
        "book_counts = df_ratings['isbn'].value_counts()\n"
        "df_ratings = df_ratings[df_ratings['user'].isin(user_counts[user_counts >= 200].index)]\n"
        "df_ratings = df_ratings[df_ratings['isbn'].isin(book_counts[book_counts >= 100].index)]\n"
        "\n"
        "df = df_ratings.merge(df_books, on='isbn', how='inner')\n"
        "df = df.drop_duplicates(['user', 'title'])\n"
        "\n"
        "book_user_matrix = df.pivot(index='title', columns='user', values='rating').fillna(0)\n"
        "matrix = csr_matrix(book_user_matrix.values)\n"
        "\n"
        "model = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')\n"
        "model.fit(matrix)\n"
        "print('matrix shape:', book_user_matrix.shape)"
    ),
    code(
        "# function to return recommended books - this will be tested\n"
        "def get_recommends(book=''):\n"
        "    distances, indices = model.kneighbors(\n"
        "        book_user_matrix.loc[book].values.reshape(1, -1), n_neighbors=6)\n"
        "    recommended_books = [\n"
        "        [book_user_matrix.index[indices[0][i]], float(distances[0][i])]\n"
        "        for i in range(5, 0, -1)\n"
        "    ]\n"
        "    return [book, recommended_books]"
    ),
    code(
        "books = get_recommends(\"Where the Heart Is (Oprah's Book Club (Paperback))\")\n"
        "print(books)"
    ),
    code(
        "# RUN THIS CELL TO TEST YOUR FUNCTION. DO NOT MODIFY CONTENTS.\n"
        "def test_book_recommendation():\n"
        "    test_pass = True\n"
        "    recommends = get_recommends(\"Where the Heart Is (Oprah's Book Club (Paperback))\")\n"
        "    if recommends[0] != \"Where the Heart Is (Oprah's Book Club (Paperback))\":\n"
        "        test_pass = False\n"
        "    recommended_books = [\"I'll Be Seeing You\", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']\n"
        "    recommended_books_dist = [0.8, 0.77, 0.77, 0.77]\n"
        "    for i in range(2):\n"
        "        if recommends[1][i][0] not in recommended_books:\n"
        "            test_pass = False\n"
        "        if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:\n"
        "            test_pass = False\n"
        "    if test_pass:\n"
        "        print('You passed the challenge! 🎉🎉🎉🎉🎉')\n"
        "    else:\n"
        "        print(\"You haven't passed yet. Keep trying!\")\n"
        "\n"
        "test_book_recommendation()"
    ),
])

# ---------------------------------------------------------------- Health costs
health = nb([
    md("# Linear Regression Health Costs Calculator\n\n"
       "Goal: Mean Absolute Error below 3500 on the test set."),
    code(
        "# Import libraries. You may or may not use all of these.\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "try:\n"
        "    %tensorflow_version 2.x\n"
        "except Exception:\n"
        "    pass\n"
        "import tensorflow as tf\n"
        "from tensorflow import keras\n"
        "from tensorflow.keras import layers\n"
        "tf.random.set_seed(42)\n"
        "np.random.seed(42)"
    ),
    code(
        "# Import data\n"
        "import os\n"
        "if os.path.exists('data/insurance.csv'):\n"
        "    dataset = pd.read_csv('data/insurance.csv')\n"
        "else:\n"
        "    if not os.path.exists('insurance.csv'):\n"
        "        !wget -q https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv\n"
        "    dataset = pd.read_csv('insurance.csv')\n"
        "dataset.tail()"
    ),
    code(
        "# Pre-processing: one-hot encode the categorical columns and split 80/20.\n"
        "dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'], drop_first=True)\n"
        "\n"
        "train_dataset = dataset.sample(frac=0.8, random_state=0)\n"
        "test_dataset = dataset.drop(train_dataset.index)\n"
        "\n"
        "train_labels = train_dataset.pop('expenses')\n"
        "test_labels = test_dataset.pop('expenses')\n"
        "\n"
        "normalizer = layers.Normalization(axis=-1)\n"
        "normalizer.adapt(np.array(train_dataset))"
    ),
    code(
        "model = keras.Sequential([\n"
        "    normalizer,\n"
        "    layers.Dense(64, activation='relu'),\n"
        "    layers.Dense(64, activation='relu'),\n"
        "    layers.Dense(1),\n"
        "])\n"
        "model.compile(optimizer=keras.optimizers.Adam(0.01), loss='mae', metrics=['mae', 'mse'])\n"
        "history = model.fit(train_dataset, train_labels, epochs=120, batch_size=32,\n"
        "                    validation_split=0.2, verbose=0)\n"
        "print('final train MAE:', history.history['mae'][-1])"
    ),
    code(
        "# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.\n"
        "# Test model by checking how well the model generalizes using the test set.\n"
        "loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)\n"
        "\n"
        "print('Testing set Mean Abs Error: {:5.2f} expenses'.format(mae))\n"
        "\n"
        "if mae < 3500:\n"
        "    print('You passed the challenge. Great job!')\n"
        "else:\n"
        "    print('The Mean Abs Error must be less than 3500. Keep trying.')\n"
        "\n"
        "# Plot predictions.\n"
        "test_predictions = model.predict(test_dataset).flatten()\n"
        "\n"
        "a = plt.axes(aspect='equal')\n"
        "plt.scatter(test_labels, test_predictions)\n"
        "plt.xlabel('True values (expenses)')\n"
        "plt.ylabel('Predictions (expenses)')\n"
        "lims = [0, 50000]\n"
        "plt.xlim(lims)\n"
        "plt.ylim(lims)\n"
        "_ = plt.plot(lims, lims)\n"
        "plt.show()"
    ),
])

# ---------------------------------------------------------------- SMS
sms = nb([
    md("# Neural Network SMS Text Classifier\n\n"
       "Binary classifier (ham / spam) built with a TextVectorization layer, "
       "an embedding and a bidirectional LSTM."),
    code(
        "import os\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import tensorflow as tf\n"
        "from tensorflow import keras\n"
        "from tensorflow.keras import layers\n"
        "import matplotlib.pyplot as plt\n"
        "tf.random.set_seed(42)\n"
        "np.random.seed(42)\n"
        "print(tf.__version__)"
    ),
    code(
        "# get data files\n"
        "if os.path.exists('data/train-data.tsv'):\n"
        "    train_file_path = 'data/train-data.tsv'\n"
        "    test_file_path = 'data/valid-data.tsv'\n"
        "else:\n"
        "    if not os.path.exists('train-data.tsv'):\n"
        "        !wget -q https://cdn.freecodecamp.org/project-data/sms/train-data.tsv\n"
        "        !wget -q https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv\n"
        "    train_file_path = 'train-data.tsv'\n"
        "    test_file_path = 'valid-data.tsv'"
    ),
    code(
        "train_data = pd.read_csv(train_file_path, sep='\\t', names=['label', 'message'])\n"
        "valid_data = pd.read_csv(test_file_path, sep='\\t', names=['label', 'message'])\n"
        "\n"
        "train_text = train_data['message'].values\n"
        "valid_text = valid_data['message'].values\n"
        "train_labels = (train_data['label'].values == 'spam').astype('float32')\n"
        "valid_labels = (valid_data['label'].values == 'spam').astype('float32')\n"
        "print(train_data.shape, valid_data.shape)"
    ),
    code(
        "max_features = 10000\n"
        "sequence_length = 120\n"
        "\n"
        "vectorizer = layers.TextVectorization(\n"
        "    max_tokens=max_features, output_mode='int',\n"
        "    output_sequence_length=sequence_length)\n"
        "vectorizer.adapt(train_text)\n"
        "\n"
        "model = keras.Sequential([\n"
        "    vectorizer,\n"
        "    layers.Embedding(max_features, 32),\n"
        "    layers.Bidirectional(layers.LSTM(32)),\n"
        "    layers.Dense(32, activation='relu'),\n"
        "    layers.Dropout(0.3),\n"
        "    layers.Dense(1, activation='sigmoid'),\n"
        "])\n"
        "model.compile(optimizer=keras.optimizers.Adam(1e-3),\n"
        "              loss='binary_crossentropy', metrics=['accuracy'])\n"
        "history = model.fit(train_text, train_labels, epochs=12, batch_size=64,\n"
        "                    validation_data=(valid_text, valid_labels), verbose=2)\n"
        "print('validation accuracy:', history.history['val_accuracy'][-1])"
    ),
    code(
        "# function to predict messages based on model\n"
        "# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])\n"
        "def predict_message(pred_text):\n"
        "    batch = tf.constant([pred_text])\n"
        "    probability = float(model.predict(batch, verbose=0)[0][0])\n"
        "    return [probability, 'spam' if probability > 0.5 else 'ham']\n"
        "\n"
        "\n"
        "pred_text = 'how are you doing today?'\n"
        "prediction = predict_message(pred_text)\n"
        "print(prediction)"
    ),
    code(
        "# Run this cell to test your function and model. Do not modify contents.\n"
        "def test_predictions():\n"
        "    test_messages = ['how are you doing today',\n"
        "                     'sale today! to stop texts call 98912460324',\n"
        "                     'i dont want to go. can we try it a different day? available sat',\n"
        "                     'our new mobile video service is live. just install on your phone to start watching.',\n"
        "                     'you have won £1000 cash! call to claim your prize.',\n"
        "                     \"i'll bring it tomorrow. don't forget the milk.\",\n"
        "                     'wow, is your arm alright. that happened to me one time too']\n"
        "\n"
        "    test_answers = ['ham', 'spam', 'ham', 'spam', 'spam', 'ham', 'ham']\n"
        "    passed = True\n"
        "\n"
        "    for msg, ans in zip(test_messages, test_answers):\n"
        "        prediction = predict_message(msg)\n"
        "        if prediction[1] != ans:\n"
        "            passed = False\n"
        "\n"
        "    if passed:\n"
        "        print('You passed the challenge. Great job!')\n"
        "    else:\n"
        "        print(\"You haven't passed yet. Keep trying.\")\n"
        "\n"
        "test_predictions()"
    ),
])

for name, notebook in [
    ("RPS.ipynb", rps),
    ("fcc_cat_dog.ipynb", catdog),
    ("fcc_book_recommendation_knn.ipynb", book),
    ("fcc_predict_health_costs_with_regression.ipynb", health),
    ("fcc_sms_text_classification.ipynb", sms),
]:
    with open(os.path.join(HERE, name), "w") as f:
        nbf.write(notebook, f)
    print("wrote", name)
