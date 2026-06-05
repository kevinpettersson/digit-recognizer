import torchvision
import torch
import torchvision.transforms as transforms

transform = transforms.ToTensor()

def get_dataloaders(batch_size=64):

    train_set = torchvision.datasets.MNIST(
        root='../../data', 
        train=True, 
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    test_set = torchvision.datasets.MNIST(
        root='../../data', 
        train=False, 
        download=True,
        transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader