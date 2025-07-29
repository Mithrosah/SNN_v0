import torch
import torch.nn as nn
from tqdm import tqdm

from model import SMNIST, MNIST
from data import mnist_loader
from utils import seed_everything

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SMNIST().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dl, valid_dl = mnist_loader()

def train(epoch):
    model.train()
    running_loss = 0.0
    acc = 0.0
    for data, target in tqdm(train_dl, desc=f'Training Epoch {epoch + 1:02d}'):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(data)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(target)
        is_correct = (torch.argmax(pred, dim=1) == target).float()
        acc += is_correct.sum().cpu()
    running_loss /= len(train_dl.dataset)
    acc /= len(train_dl.dataset)
    return running_loss, acc

def evaluate(epoch):
    model.eval()
    running_loss = 0.0
    acc = 0.0
    with torch.no_grad():
        for data, target in tqdm(valid_dl, desc=f'Validation Epoch {epoch + 1:02d}'):
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss = loss_fn(pred, target)
            running_loss += loss.item() * len(target)
            is_correct = (torch.argmax(pred, dim=1) == target).float()
            acc += is_correct.sum().cpu()
    running_loss /= len(valid_dl.dataset)
    acc /= len(valid_dl.dataset)
    return running_loss, acc

def main():
    epochs = 20
    for epoch in range(epochs):
        train_loss, train_acc = train(epoch)
        valid_loss, valid_acc = evaluate(epoch)
        print(f'Epoch {epoch + 1}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, '
              f'train_acc: {train_acc:.4f}, valid_acc: {valid_acc:.4f}')

if __name__ == '__main__':
    seed_everything(seed=42)
    main()