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
import sys
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler

def check_gpu():
    if torch.cuda.is_available():
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}") 
        return "cuda"
    else:
        print("GPU não encontrada. O programa será encerrado.")
        sys.exit()  

device = check_gpu()

torch.manual_seed(42)
np.random.seed(42)

data_dir = 'imgs/'
num_classes = 78
epochs = 125

# Hiperparâmetros
best_lr = 0.00044220953014257724
best_optimizer_name = "Adam"
best_batch_size = 128
best_dropout_rate = 0.07609417454082278
best_n_units_fc1 = 4096
best_n_units_fc2 = 1792

# Função de normalização e aumento de dados (Data Augmentation)
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

# Carregar o conjunto de dados de teste
test_dataset = ImageFolder(os.path.join(data_dir), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)

# Contar a distribuição de classes no dataset de treinamento
class_counts = Counter([label for _, label in train_dataset])
class_weights = [1.0 / class_counts[i] for i in range(num_classes)]

# Calcular pesos para cada amostra no dataset de treinamento
sample_weights = [class_weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Atualizar o DataLoader com o sampler
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=best_batch_size, shuffle=False)


num_classes = len(os.listdir("./imgs"))
print(f"{num_classes} classes encontradas")

# Carregar o modelo ResNet50 pré-treinado
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, best_n_units_fc1),
    nn.ReLU(inplace=True),
    nn.Dropout(best_dropout_rate),
    nn.Linear(best_n_units_fc1, best_n_units_fc2),
    nn.ReLU(inplace=True),
    nn.Dropout(best_dropout_rate),
    nn.Linear(best_n_units_fc2, num_classes)
)

model.to(device)

# Restante do código permanece o mesmo para otimização, treinamento, validação e teste


# Definir o otimizador com os melhores parâmetros encontrados
if best_optimizer_name == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
elif best_optimizer_name == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=best_lr, momentum=0.9)
elif best_optimizer_name == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=best_lr)
elif best_optimizer_name == 'Adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=best_lr)
elif best_optimizer_name == 'Adamax':
    optimizer = optim.Adamax(model.parameters(), lr=best_lr)

# Adicionar um scheduler para reduzir a taxa de aprendizado durante o treinamento

# Função de perda para o modelo final
final_criterion = nn.CrossEntropyLoss()

# Função de treino
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

# Iniciar o treinamento do modelo com as mudanças aplicadas
print("\n\n INICIANDO TREINAMENTO DO MODELO FINAL \n\n")

train_loss_list, val_loss_list, train_accuracy_list, val_accuracy_list = train_model(
    model, final_criterion, optimizer, train_loader, val_loader, epochs=epochs
)

# Avaliar o modelo no conjunto de teste
test_model(model, test_loader)

# Plote e salve os gráficos de acurácia e perda
plt.figure(figsize=(12, 6))

# Gráfico de Acurácia
plt.subplot(1, 2, 1)
plt.plot(train_accuracy_list, label='Train Accuracy')
plt.plot(val_accuracy_list, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Gráfico de Perda
plt.subplot(1, 2, 2)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()


os.makedirs('./plots', exist_ok=True)

# Salvar os gráficos em um arquivo
plt.savefig('./plots/densenet.png')
plt.show()

os.makedirs('./models', exist_ok=True)

# Salvar o modelo treinado
model_save_path = './models/densenet.pth'
torch.save(model.state_dict(), model_save_path)

print(f'Modelo salvo em {model_save_path}')
