import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
import os

ds = tf.data.Dataset.list_files('archive/files/*/*.jpg', shuffle=False)

IMAGE_SIZE = 516
BATCH = 32
CHANNELS = 3
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "archive/files",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH
)

class_ = dataset.class_names

for img, label in dataset.take(1):
    print(img[0].shape)
    plt.figure(figsize=(10, 10))
    for i in range(6):
        plt.subplot(3, 3, i + 1)
        plt.title(class_[label[i]])
        plt.imshow(img[i].numpy().astype("uint8"))

def split_train_test_val(ds, train_split=0.6, test_split=0.2, val_split=0.2, shuffle=False):
    ds_size = len(ds)
    train_size = round(train_split * ds_size)
    test_size = round(test_split * ds_size)
    val_size = round(val_split * ds_size)
    if shuffle:
        ds = ds.shuffle()

    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size).take(test_size)
    val_ds = ds.skip(train_size).skip(test_size)
    return train_ds, test_ds, val_ds

train_ds, test_ds, val_ds = split_train_test_val(ds=dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

image_preprocessing = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE,
                            interpolation='bilinear',
                            crop_to_aspect_ratio=False),
    layers.Rescaling(1. / 255)
])

input_size = (BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
model = models.Sequential([
    image_preprocessing,
    layers.Conv2D(32, (8, 8), activation='relu', input_shape=input_size),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    layers.Conv2D(64, kernel_size=(4, 4), activation='relu'),
    layers.Dropout(.2),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(len(class_), activation='softmax')
])

model.build(input_shape=input_size)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

EPOCHS = 10
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, verbose=1)

# Salvar a acurácia e a perda em um arquivo de texto
with open('metrics.txt', 'w') as file:
    file.write("Epoch\tTrain Loss\tTrain Accuracy\tValidation Loss\tValidation Accuracy\n")
    for epoch in range(EPOCHS):
        train_loss = history.history['loss'][epoch]
        train_accuracy = history.history['accuracy'][epoch]
        val_loss = history.history['val_loss'][epoch]
        val_accuracy = history.history['val_accuracy'][epoch]
        file.write(f"{epoch + 1}\t{train_loss}\t{train_accuracy}\t{val_loss}\t{val_accuracy}\n")

# Plotar o gráfico de acurácia e perda
epochs = list(range(EPOCHS))
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label="Train Accuracy")
plt.plot(epochs, history.history['val_accuracy'], label="Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label="Train Loss")
plt.plot(epochs, history.history['val_loss'], label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig('training_metrics.png')

# Salvar o modelo treinado em um arquivo
model.save('trained_model.h5')

score = model.evaluate(test_ds)
print(score)

for img, label in test_ds.take(1):
    plt.figure(figsize=(10, 10))
    for i in range(6):
        plt.subplot(3, 3, i + 1)
        pred = model.predict(img.numpy())
        plt.title(f"Actual : {class_[label.numpy()[i]]} \n Predicted : {class_[np.argmax(pred[i])]}")
        plt.imshow(img[i].numpy())
        plt.axis("off")
