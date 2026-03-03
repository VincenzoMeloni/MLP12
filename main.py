from data.mnist_loader import get_mnist_loaders

train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=32)

for images, labels in train_loader:
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    break