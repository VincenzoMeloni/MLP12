from data.mnist_loader import get_mnist_loaders
from src.model.network import MLP
from src.training.trainer import train_model
from src.utils.plot import plot_losses, plot_accuracies

# caricamento dati
train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=32)

#creazione modello
model = MLP(hidden_size=128)

# Training
train_losses, val_losses, train_accs, val_accs = train_model(
    model,
    train_loader,
    val_loader,
    epochs=5,
    lr=0.01
)

plot_losses(train_losses, val_losses, title="MLP - Hidden 128 - Loss")
plot_accuracies(train_accs, val_accs, title="MLP - Hidden 128 - Accuracy")