from data.mnist_loader import get_mnist_loaders

train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=32)

print("Training samples:", len(train_loader.dataset))
print("Validation samples:", len(val_loader.dataset))
print("Test samples:", len(test_loader.dataset))

print("Train indices:", len(train_loader.dataset.indices))
print("Val indices:", len(val_loader.dataset.indices))

train_indices = set(train_loader.dataset.indices)
val_indices = set(val_loader.dataset.indices)

overlap = train_indices.intersection(val_indices)

print("Overlap size:", len(overlap))