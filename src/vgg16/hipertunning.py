import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import optuna
import sys

def check_gpu():
    if torch.cuda.is_available():
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}") 
        return "cuda"
    else:
        print("GPU não encontrada. O programa será encerrado.")
        sys.exit()  

device = check_gpu()

data_dir = 'imgs/'
batch_size = 64
num_classes = 78
epochs = 200

def image_normalizer():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

transform = image_normalizer()

# Carregar o conjunto de dados de treinamento e validação

train_dataset = ImageFolder(os.path.join(data_dir), transform=transform)
train_size = int(0.7 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Carregar os dados usando DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Carregar o conjunto de dados de teste
test_dataset = ImageFolder(os.path.join(data_dir), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Função para treinar o modelo
def train_model(model, criterion, optimizer, train_loader, val_loader, epochs):
    train_loss_list, val_loss_list = [], []
    train_accuracy_list, val_accuracy_list = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=True)
        loop.set_description(f'Epoch {epoch+1}/{epochs}')

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zerar os gradientes
            optimizer.zero_grad()

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            # Estatísticas de acurácia
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        val_loss, val_accuracy = validate_model(model, val_loader, criterion)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list

# Função para validar o modelo
def validate_model(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    return val_loss, val_accuracy

# Função para avaliar o modelo no conjunto de teste
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

# Definir a função de objetivo para otimização
import os

def objective(trial):
    # Hiperparâmetros a serem otimizados
    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adamax'])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5) 
    n_units_fc1 = trial.suggest_int('n_units_fc1', 1024, 4096, step=512)
    n_units_fc2 = trial.suggest_int('n_units_fc2', 512, 2048, step=256)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_classes = len(os.listdir("./imgs"))

    # Carregar o modelo VGG16 pré-treinado
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Congelar as camadas convolucionais
    for param in model.features.parameters():
        param.requires_grad = False

    # Modificar a camada classifier
    model.classifier = nn.Sequential(
        nn.Linear(25088, n_units_fc1),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(n_units_fc1, n_units_fc2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(n_units_fc2, num_classes)
    )

    model.to(device)

    # Definir o otimizador
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
    elif optimizer_name == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=lr)

    # Função de perda
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list = train_model(
        model, criterion, optimizer, train_loader, val_loader, epochs
    )

    # Salvar resultados em um arquivo txt
    val_accuracy = val_accuracy_list[-1]
    results_path = "experiment_results.txt"
    
    with open(results_path, 'a') as f:
        f.write(f"Trial: {trial.number}, LR: {lr}, Optimizer: {optimizer_name}, Batch size: {batch_size}, "
                f"Dropout rate: {dropout_rate}, FC1 units: {n_units_fc1}, FC2 units: {n_units_fc2}, "
              f"Val Accuracy: {val_accuracy}\n")

    # Retornar a acurácia de validação da última época para otimização
    return val_accuracy

print("\n\n INICIANDO OTIMIZACAO \n\n")

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Após a otimização, salvar o melhor resultado
best_trial = study.best_trial
best_results_path = "best_experiment_result.txt"

with open(best_results_path, 'w') as f:
    f.write(f"Best Trial: {best_trial.number}, Val Accuracy: {best_trial.value}, "
            f"Params: {best_trial.params}\n")


print("="*20)
print("Melhores valores:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print(f"  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")