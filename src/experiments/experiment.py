import os
import torch
from src.model.network import MLP
from src.training.trainer import train_model
from src.utils.plot import plot_losses, plot_accuracies
import csv

def run_experiment(train_loader,val_loader,test_loader,training_mode="minibatch",hidden_size=128,epochs=10,lr=0.01):

    print(f"\n========== Running {training_mode} experiment ==========")
    print(f"Hidden size: {hidden_size}")

    # Crea modello
    model = MLP(hidden_size=hidden_size)

    # Training
    train_losses, val_losses, train_accs, val_accs = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=lr
    )

    # Test Set Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total

    print(f"\nTest Accuracy: {test_accuracy:.2f}%")

    # Salvataggio modello
    os.makedirs("models", exist_ok=True)

    batch_size = train_loader.batch_size
    model_name = f"model_{training_mode}_bs{batch_size}_h{hidden_size}.pth"

    torch.save(model.state_dict(), os.path.join("models", model_name))

    print(f"Model saved: models/{model_name}")

    # Salvataggio grafici
    os.makedirs("charts", exist_ok=True)

    loss_filename = f"loss_{training_mode}_bs{batch_size}_h{hidden_size}.png"
    acc_filename  = f"acc_{training_mode}_bs{batch_size}_h{hidden_size}.png"

    # Plot automatici
    plot_losses(train_losses, val_losses,title=f"{training_mode} - Hidden {hidden_size}",filename=loss_filename)

    plot_accuracies(train_accs, val_accs,title=f"{training_mode} - Hidden {hidden_size}",filename=acc_filename)

    print("Experiment finished.\n")

    # Salvataggio risultati in CSV
    csv_file = "result.csv"
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a",newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Training Mode",
                "Hidden Size",
                "Batch Size",
                "Epochs",
                "Learning Rate",
                "Test Accurancy"
            ])
        
        writer.writerow([
            training_mode,
            hidden_size,
            train_loader.batch_size,
            epochs,
            lr,
            f"{test_accuracy:.2f}"
        ])

    return test_accuracy