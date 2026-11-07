"""Build the final cat/dog notebook (MobileNetV2 transfer learning)."""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


cells = [
    md("# Cat and Dog Image Classifier\n\n"
       "Solution for the freeCodeCamp *Machine Learning with Python* project.\n\n"
       "A MobileNetV2 base pre-trained on ImageNet is used as a feature extractor, "
       "followed by a small classifier head. The top 30 layers are then fine-tuned. "
       "Horizontal-flip test-time augmentation is applied on the 50 test images.\n\n"
       "Target: at least 63% of the test images correctly classified."),
    code(
        "try:\n"
        "    %tensorflow_version 2.x\n"
        "except Exception:\n"
        "    pass\n"
        "import tensorflow as tf\n"
        "from tensorflow.keras.models import Sequential\n"
        "from tensorflow.keras.layers import (Dense, Conv2D, Flatten, Dropout,\n"
        "                                     MaxPooling2D, GlobalAveragePooling2D)\n"
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n"
        "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau\n"
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
        "epochs = 8\n"
        "IMG_HEIGHT = 150\n"
        "IMG_WIDTH = 150\n"
        "print(total_train, total_val, total_test)"
    ),
    code(
        "# 3\n"
        "preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input\n"
        "\n"
        "train_image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)\n"
        "validation_image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)\n"
        "test_image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)\n"
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
        "            ax.imshow(np.clip(img * 0.5 + 0.5, 0, 1) if img.min() < 0 else img)\n"
        "            ax.axis('off')\n"
        "    else:\n"
        "        for img, probability, ax in zip(images_arr, probabilities, axes):\n"
        "            ax.imshow(np.clip(img * 0.5 + 0.5, 0, 1) if img.min() < 0 else img)\n"
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
        "# 5 - data augmentation\n"
        "train_image_generator = ImageDataGenerator(\n"
        "    preprocessing_function=preprocess_input,\n"
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
        "# 7 - MobileNetV2 base + classifier head\n"
        "base_model = tf.keras.applications.MobileNetV2(\n"
        "    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')\n"
        "base_model.trainable = False\n"
        "\n"
        "model = Sequential([\n"
        "    base_model,\n"
        "    GlobalAveragePooling2D(),\n"
        "    Dropout(0.3),\n"
        "    Dense(1, activation='sigmoid'),\n"
        "])\n"
        "model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),\n"
        "              loss='binary_crossentropy', metrics=['accuracy'])\n"
        "model.summary()"
    ),
    code(
        "# 8 - stage 1: train the head with the base frozen\n"
        "history = model.fit(\n"
        "    train_data_gen,\n"
        "    steps_per_epoch=total_train // batch_size,\n"
        "    epochs=epochs,\n"
        "    validation_data=val_data_gen,\n"
        "    validation_steps=total_val // batch_size,\n"
        "    callbacks=[EarlyStopping(monitor='val_accuracy', patience=3,\n"
        "                             restore_best_weights=True, verbose=1)],\n"
        "    verbose=2)"
    ),
    code(
        "# 8b - stage 2: fine-tune the top 30 layers of MobileNetV2\n"
        "base_model.trainable = True\n"
        "for layer in base_model.layers[:-30]:\n"
        "    layer.trainable = False\n"
        "for layer in base_model.layers:\n"
        "    if isinstance(layer, tf.keras.layers.BatchNormalization):\n"
        "        layer.trainable = False\n"
        "\n"
        "model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),\n"
        "              loss='binary_crossentropy', metrics=['accuracy'])\n"
        "history = model.fit(\n"
        "    train_data_gen,\n"
        "    steps_per_epoch=total_train // batch_size,\n"
        "    epochs=8,\n"
        "    validation_data=val_data_gen,\n"
        "    validation_steps=total_val // batch_size,\n"
        "    callbacks=[EarlyStopping(monitor='val_accuracy', patience=3,\n"
        "                             restore_best_weights=True, verbose=1),\n"
        "               ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,\n"
        "                                 patience=1, min_lr=1e-7, verbose=1)],\n"
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
        "# 10 - predictions on the 50 test images (with horizontal-flip TTA)\n"
        "test_data_gen.reset()\n"
        "test_images = []\n"
        "for i in range(len(test_data_gen)):\n"
        "    test_images.append(test_data_gen[i])  # batch (1, H, W, 3)\n"
        "test_images = np.vstack(test_images)\n"
        "flipped = np.ascontiguousarray(test_images[:, :, ::-1, :])\n"
        "probabilities = (model.predict(test_images, verbose=0).flatten()\n"
        "                 + model.predict(flipped, verbose=0).flatten()) / 2.0\n"
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
]

notebook = nbf.v4.new_notebook()
notebook["cells"] = cells
notebook["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
    "colab": {"provenance": []},
}
with open(os.path.join(HERE, "fcc_cat_dog.ipynb"), "w") as f:
    nbf.write(notebook, f)
print("wrote fcc_cat_dog.ipynb")
