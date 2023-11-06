import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregando os dados
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
class_names = dataset.class_names

# Função para dividir o dataset em treinamento, teste e validação
def split_train_test_val(ds, train_split=0.6, test_split=0.2, val_split=0.2, shuffle=False):
    ds_size = len(ds)
    train_size = round(train_split * ds_size)
    test_size = round(test_split * ds_size)
    val_size = round(val_split * ds_size)
    if shuffle:
        ds = ds.shuffle(ds_size)
    train_ds = ds.take(train_size)
    test_ds = ds.skip(train_size).take(test_size)
    val_ds = ds.skip(train_size + test_size)
    return train_ds, test_ds, val_ds

train_ds, test_ds, val_ds = split_train_test_val(ds=dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

image_preprocessing = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMAGE_SIZE, IMAGE_SIZE, interpolation='bilinear', crop_to_aspect_ratio=False),
    layers.Rescaling(1. / 255)
])

input_size = (BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

# Defina a CNN como um modelo para extração de características
cnn_model = models.Sequential([
    image_preprocessing,
    layers.Conv2D(32, (8, 8), activation='relu', input_shape=input_size),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
    layers.Conv2D(64, kernel_size=(4, 4), activation='relu'),
    layers.Dropout(.2),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.Flatten()
])

cnn_model.compile(optimizer="adam")

# Extraia as características das imagens de treinamento, teste e validação
def extract_features(model, data):
    features = []
    for batch in data:
        batch_features = model.predict(batch)
        features.append(batch_features)
    return np.vstack(features)

train_features = extract_features(cnn_model, train_ds)
test_features = extract_features(cnn_model, test_ds)
val_features = extract_features(cnn_model, val_ds)

# Treine um modelo Random Forest com as características extraídas
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_features, train_ds.labels)

# Faça previsões no conjunto de teste
predictions = rf_model.predict(test_features)

# Calcule a precisão
accuracy = accuracy_score(test_ds.labels, predictions)
print("Accuracy: ", accuracy)

# Salvar os resultados em um arquivo TXT
with open('cnn_as_fe/results.txt', 'w') as f:
    f.write(f"Test Accuracy: {accuracy}\n")

# Salvar o modelo treinado
import joblib
joblib.dump(rf_model, 'cnn_as_fe/random_forest_model.pkl')
