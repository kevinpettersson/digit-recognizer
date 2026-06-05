import torch
import torch.optim as optim
import torch.nn as nn

torch.manual_seed(41)

from model.neural_network import Model
from data.dataloader import get_dataloaders

model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)

train_loader, test_loader = get_dataloaders()

def train():
    model.train()

    for epoch in range(5):

        for i, (images, labels) in enumerate(train_loader):

            # 1. Forward pass
            outputs = model(images)

            # 2. Calculate loss
            loss = criterion(outputs, labels)

            # 3. Clear old gradients
            optimizer.zero_grad()

            # 4. Backpropagation
            loss.backward()

            # 5. Update weights
            optimizer.step()

            # print statistics
            if i % 100 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

    print('Finished Training')
    PATH = './mnist_model.pt'
    torch.save(model.state_dict(), PATH)

def test():
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for (images, labels) in test_loader:

            # 
            outputs = model(images)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

if __name__ == "__main__":
    train()
    test()