# Imports necesarios para el módulo
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def display_digits(x_train, y_train):
    """
    Método para visualizar 4 dígitos aleatorios y sus etiquetas.
    Crea una figura de 2x2 subplots donde cada subplot muestra:
    - Una imagen aleatoria del conjunto de entrenamiento 
    - El número de ejemplo y su etiqueta correspondiente
    Los dígitos se muestran en escala de grises invertida (gray_r)
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(2):
        for j in range(2):
            num = np.random.randint(0, x_train.shape[0])
            image = x_train[num,:,:]
            label = y_train[num]
            axs[i, j].imshow(image, cmap=plt.get_cmap('gray_r'))
            axs[i, j].set_title(f'Example: {num};  Label: {label}')
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()
    
def display_split_pie_chart(x_train, x_val, x_test):
    """
    Función que muestra un gráfico de pie que representa la distribución 
    de los conjuntos de datos (train, validation y test)
    """
    plt.figure(figsize=(10, 5))
    sizes = [x_train.shape[0], x_val.shape[0], x_test.shape[0]]
    labels = ['Train', 'Validation', 'Test']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribución de los conjuntos de datos')
    plt.show()
    
def display_as_signal(x_train):
    """
    Función que muestra un gráfico con dos subplots:
    - El primer subplot muestra el primer dígito MNIST como imagen
    - El segundo subplot muestra el primer dígito MNIST como señal
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    digit_image = x_train[0].numpy().reshape(28, 28)
    ax1.imshow(digit_image, cmap='gray_r')
    ax1.set_title('Primer dígito MNIST como imagen')
    ax1.axis('off')

    ax2.plot(x_train[0].numpy())
    ax2.set_title('Primer dígito MNIST como señal')
    ax2.set_xlabel('Posición del pixel')
    ax2.set_ylabel('Intensidad del pixel')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    
def display_loss_and_accuracy(loss_epoch_tr, loss_epoch_val, acc_epoch_tr, acc_epoch_val, test_acc, n_epochs):
    """
    Función que muestra un gráfico con dos subplots:
    - El primer subplot muestra el loss por epoch
    - El segundo subplot muestra la precisión por epoch
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(range(n_epochs), loss_epoch_tr, label='Entrenamiento')
    ax1.plot(range(n_epochs), loss_epoch_val, label='Validación')
    ax1.legend(loc='upper right')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch #')
    ax1.set_ylabel('Loss')

    ax2.plot(range(n_epochs), acc_epoch_tr, label='Entrenamiento')
    ax2.plot(range(n_epochs), acc_epoch_val, label='Validación')
    ax2.axhline(y=test_acc, color='red', linestyle='--', label='Test')
    ax2.legend(loc='lower right')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch #')
    ax2.set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()
    
def sanity_check(x_test, idx, true_class, prediction, predicted_class):
    """
    Función que muestra un gráfico con dos subplots:
    - El primer subplot muestra la imagen del dígito real
    - El segundo subplot muestra las probabilidades para cada clase
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    img_reshaped = x_test[idx].reshape(28, 28)
    plt.imshow(img_reshaped.numpy(), cmap='gray')
    plt.title(f'Dígito real: {true_class.item()}')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    probs = torch.nn.functional.softmax(prediction[0], dim=0)
    plt.bar(range(10), probs.numpy())
    plt.title(f'Predicción del modelo: {predicted_class.item()}\nProbabilidades por clase')
    plt.xlabel('Clase')
    plt.ylabel('Probabilidad')
    
    plt.tight_layout()
    plt.show()
    
###############
# Notebook II #
###############


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device):
    """
    Evalúa el modelo en el conjunto de datos proporcionado y devuelve las predicciones
    y los valores reales para su posterior análisis
    """
    print("[INFO]: Evaluando red neuronal...")
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return all_predictions, all_targets

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs, test_acc=None):
    """
    Visualiza las curvas de entrenamiento mostrando la evolución del loss y 
    la precisión durante el entrenamiento
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accs, label="Train Accuracy")
    plt.plot(range(num_epochs), val_accs, label="Validation Accuracy")
    if test_acc is not None:
        plt.axhline(y=test_acc, color='red', linestyle='--', label='Test Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_learning_rate(lrs, num_epochs):
    """
    Visualiza la evolución del learning rate a lo largo de las épocas
    de entrenamiento
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), lrs)
    plt.title("Learning Rate Decay")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.yscale('log')
    plt.grid(True)
    plt.show()

################
# Notebook III #
################


def train_epoch(model: nn.Module, device: torch.device, train_loader: DataLoader, 
                criterion, optimizer, scheduler):
    """
    Entrena una época de la red neuronal y devuelve las métricas de entrenamiento.
    
    Args:
        model: Modelo de red neuronal a entrenar
        device: Dispositivo donde se realizará el entrenamiento (CPU/GPU)
        train_loader: DataLoader con los datos de entrenamiento
        criterion: Función de pérdida a utilizar
        optimizer: Optimizador para actualizar los pesos
        scheduler: Scheduler para ajustar el learning rate
        
    Returns:
        train_loss: Pérdida promedio en el conjunto de entrenamiento
        train_acc: Precisión en el conjunto de entrenamiento (%)
        current_lr: Learning rate actual después del scheduler
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss /= len(train_loader)
    train_acc = 100. * correct / total

    # Aplicar el scheduler después de cada época
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    return train_loss, train_acc, current_lr

def eval_epoch(model: nn.Module, device: torch.device, val_loader: DataLoader, 
               criterion):
    """
    Evalúa el modelo en el conjunto de validación.
    
    Args:
        model: Modelo de red neuronal a evaluar
        device: Dispositivo donde se realizará la evaluación (CPU/GPU)
        val_loader: DataLoader con los datos de validación
        criterion: Función de pérdida a utilizar
        
    Returns:
        val_loss: Pérdida promedio en el conjunto de validación
        val_acc: Precisión en el conjunto de validación (%)
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc