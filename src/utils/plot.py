import matplotlib.pyplot as plt
import os

def _ensure_chart_folder():
    folder = "charts"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

# Loss Plot
def plot_losses(train_losses, val_losses,title="Loss Curve",filename="loss.png"):

    folder = _ensure_chart_folder()
    save_path = os.path.join(folder, filename)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()

    print(f"Loss chart saved at: {save_path}")

# Accuracy Plot
def plot_accuracies(train_accuracies,val_accuracies,title="Accuracy Curve",filename="accuracy.png"):

    folder = _ensure_chart_folder()
    save_path = os.path.join(folder, filename)

    plt.figure()
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()

    print(f"Accuracy chart saved at: {save_path}")