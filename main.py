from data.mnist_loader import get_mnist_loaders
from src.experiments.experiment import run_experiment

# PARAMETRI 
hidden_sizes = [64, 128, 256]

training_modes = [
    ("online", [1]),
    ("batch", [None]),
    ("minibatch", [16, 32, 64])
]

epochs = 30
lr = 0.01

for hidden_size in hidden_sizes:

    print("\n" + "="*60)
    print(f"START EXPERIMENTS - Hidden Size: {hidden_size}")
    print("="*60)

    for mode, batch_sizes in training_modes:

        for batch_size in batch_sizes:

            print(f"\n[Mode]: {mode} | [Batch Size]: {batch_size}")

            if mode == "batch":
                train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=60000)
            else:
                train_loader, val_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

            run_experiment(
                train_loader,
                val_loader,
                test_loader,
                training_mode=mode,
                hidden_size=hidden_size,
                epochs=epochs,
                lr=lr
            )

print("\nALL EXPERIMENTS COMPLETED")