import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_dataloaders(data_dir='data', batch_size=32, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB normalization
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'training'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'testing'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes


def get_class_names(data_dir='data/training'):
    dataset = datasets.ImageFolder(data_dir)
    return dataset.classes
