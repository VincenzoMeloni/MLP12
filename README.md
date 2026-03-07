# ML-PROJECT

---

**P12)** _Use the raw MNIST images as input for a 10-class classification task. Construct a dataset of N input–label pairs and split it into training and test sets (at least 10,000 samples for training and 2,500 for testing).
Train the models using standard gradient descent.
Study the learning behavior of a neural network with a single hidden layer by varying the training modality: online, batch, and mini-batch learning. Perform this analysis for at least three different hidden-layer sizes while keeping the activation functions fixed. For each configuration, evaluate the learning dynamics (epochs to convergence, training and validation error curves, test accuracy). Additionally, examine the impact of the mini-batch size._

---

## How to Install

### 1. Clone the repository

Open **Command Prompt** and run:

    git clone <repository_url>

### 2. Create a Virtual Environment

Navigate to the **P12-PROJECT** directory and run:

    python -m venv venv

### 3. Activate the Virtual Environment

In the same directory, run:

**Windows**

    .\venv\Scripts\activate

**macOS / Linux**

    source venv/bin/activate

### 4. Install Dependencies

With the virtual environment activated, run:

    python -m pip install -r requirements.txt

### 5. Start Training

To start training, run:

    python main.py

### 6. Check Prediction (optional)

Change the `image_path` in `predict.py` (test images are located in the `images` directory) and run:

    python predict.py

**Note:** The `predict.py` script was created quickly to test whether the model can predict images outside the MNIST dataset. The test images were drawn using Paint. Feel free to modify the script or use new images to improve the results.
