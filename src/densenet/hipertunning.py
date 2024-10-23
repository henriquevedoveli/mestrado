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
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler

# Verifica se GPU está disponível
def check_gpu():
    if torch.cuda.is_available():
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}") 
        return "cuda"
    else:
        print("GPU não encontrada. O programa será encerrado.")
        sys.exit()  

device = check_gpu()

data_dir = 'imgs/'
batch_size = 128
num_classes = 78
epochs = 200

torch.manual_seed(42)
np.random.seed(42)

# Função para normalizar as imagens
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

# Contar a distribuição de classes no dataset de treinamento
class_counts = Counter([label for _, label in train_dataset])
class_weights = [1.0 / class_counts[i] for i in range(num_classes)]

# Calcular pesos para cada amostra no dataset de treinamento
sample_weights = [class_weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Atualizar o DataLoader com o sampler
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
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
            loop.set_postfix(loss=running_loss / (total // len(labels)), 
                             accuracy=100 * correct / total)

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

# Função para otimizar o modelo usando Optuna
def objective(trial):
    # Hiperparâmetros a serem otimizados
    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adamax'])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5) 
    n_units_fc1 = trial.suggest_int('n_units_fc1', 1024, 4096, step=512)
    n_units_fc2 = trial.suggest_int('n_units_fc2', 512, 2048, step=256)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Usar DenseNet em vez de ResNet
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    # Congelar as camadas convolucionais
    for param in model.parameters():
        param.requires_grad = False

    # Modificar a última camada (fully connected layer) para o número de classes (num_classes)
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, n_units_fc1), 
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(n_units_fc1, n_units_fc2),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(n_units_fc2, num_classes)
    )

    model.to(device)

    # Configuração do otimizador com os hiperparâmetros sugeridos
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
    results_path = "experiment_results_densenet_121.txt"
    
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
best_results_path = "best_experiment_result_densenet_121.txt"

with open(best_results_path, 'w') as f:
    f.write(f"Best Trial: {best_trial.number}, Val Accuracy: {best_trial.value}, "
            f"Parameters: {best_trial.params}\n")

print(f"\nMelhores hiperparâmetros: {best_trial.params}")
print(f"Acurácia de validação: {best_trial.value}")

# Teste final do modelo otimizado com os melhores parâmetros
best_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
best_model.classifier = nn.Sequential(
    nn.Linear(best_model.classifier.in_features, best_trial.params['n_units_fc1']),
    nn.ReLU(inplace=True),
    nn.Dropout(best_trial.params['dropout_rate']),
    nn.Linear(best_trial.params['n_units_fc1'], best_trial.params['n_units_fc2']),
    nn.ReLU(inplace=True),
    nn.Dropout(best_trial.params['dropout_rate']),
    nn.Linear(best_trial.params['n_units_fc2'], num_classes)
)

best_model.to(device)

if best_trial.params['optimizer'] == 'Adam':
    best_optimizer = optim.Adam(best_model.parameters(), lr=best_trial.params['lr'])
elif best_trial.params['optimizer'] == 'SGD':
    best_optimizer = optim.SGD(best_model.parameters(), lr=best_trial.params['lr'], momentum=0.9)
elif best_trial.params['optimizer'] == 'RMSprop':
    best_optimizer = optim.RMSprop(best_model.parameters(), lr=best_trial.params['lr'])
elif best_trial.params['optimizer'] == 'Adagrad':
    best_optimizer = optim.Adagrad(best_model.parameters(), lr=best_trial.params['lr'])
elif best_trial.params['optimizer'] == 'Adamax':
    best_optimizer = optim.Adamax(best_model.parameters(), lr=best_trial.params['lr'])

# Treinamento final do modelo otimizado
epochs = 10
train_model(best_model, criterion, best_optimizer, train_loader, val_loader, epochs)

# Avaliar o modelo no conjunto de teste
test_model(best_model, test_loader)

# Visualizar as perdas e acurácias de treino e validação
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_list, label='Train Accuracy')
plt.plot(val_accuracy_list, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
