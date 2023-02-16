import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T 

import matplotlib.pyplot as plt

train_mnist = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=T.Compose([
        T.ToTensor(),
        T.Resize((32, 32))
    ])
)

test_mnist = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=T.Compose([
        T.ToTensor(),
        T.Resize((32, 32))
    ])
)

train_dataloader = DataLoader(train_mnist, batch_size=560, shuffle=True)
test_dataloader = DataLoader(test_mnist, batch_size=1, shuffle=True)

global_device = "cpu"
global_epochs = 1

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.extractor_features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Sigmoid(),
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, x):
        x = self.extractor_features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        probs = nn.Softmax(dim=1)(x)

        return x, probs

def get_accuracy(model, dataloader, device):
    correct_pred = 0
    n = len(dataloader.dataset)

    with torch.no_grad():
        model.eval()

        for x, y_true in dataloader:

            x = x.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(x)
            _, pred_labels = torch.max(y_prob, 1)

            correct_pred += (pred_labels == y_true).sum()

    return float(correct_pred) / n

def train(train_loader, model, criterion, optimiser, device):
    model.train()
    running_loss = 0

    for x, y_true in train_loader:
        x = x.to(device)
        y_true = y_true.to(device)

        y_hat, _ = model(x)
        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * x.size(0)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimiser, epoch_loss

def validate(test_loader, model, criterion, device):
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for x, y_true in test_loader:
            x = x.to(device)
            y_true = y_true.to(device)

            y_hat, _ = model(x)
            loss = criterion(y_hat, y_true)
            running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(test_loader.dataset)
    return model, epoch_loss

def train_loop(model, criterion, optimiser, train_loader, test_loader, epochs, device):
    for epoch in range(epochs):
        model, optimiser, train_loss = train(train_loader, model, criterion, optimiser, device)

        with torch.no_grad():
            model, test_loss = validate(test_loader, model, criterion, device)

            train_acc = get_accuracy(model, train_loader, device)
            test_acc = get_accuracy(model, test_loader, device)

            print(f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {test_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * test_acc:.2f}')
            
    return model, optimiser

model = LeNet().to(global_device)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

model, optimiser = train_loop(model, criterion, optimiser, train_dataloader, test_dataloader, global_epochs, global_device)

x, y = next(iter(test_dataloader))
plt.imshow(x.view(32, 32), cmap="gray")
plt.show()

_, pred = model(x.to(global_device))
print(f"El valor numerico del input es: {torch.argmax(pred.reshape(10))}")