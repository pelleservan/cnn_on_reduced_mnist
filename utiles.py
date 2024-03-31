import matplotlib.pyplot as plt

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TESTSET = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
TESTLOADER = DataLoader(TESTSET, batch_size=64, shuffle=False)

def eval_model(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in TESTLOADER:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return 100 * accuracy

def classwise_accuracy(model, ax):
    model.eval()
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    with torch.no_grad():
        for images, labels in TESTLOADER:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    accuracies = [100 * class_correct[i] / class_total[i] for i in range(10)]
    digits = list(range(10))
    
    ax.bar(digits, accuracies, color='skyblue')
    ax.set_xlabel('Digit')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classwise Accuracy')
    
    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 1, f'{acc:.2f}%', ha='center')
    
    ax.set_xticks(digits)

def confusion_matrix_plot(model, ax):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in TESTLOADER:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10), ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
