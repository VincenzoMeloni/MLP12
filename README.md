# ML-PROJECT

---

**P12)** _Use the raw MNIST images as input for a 10-class classification task. Construct a dataset of N input–label pairs and split it into training and test sets (at least 10,000 samples for training and 2,500 for testing).
Train the models using standard gradient descent.
Study the learning behavior of a neural network with a single hidden layer by varying the training modality: online, batch, and mini-batch learning. Perform this analysis for at least three different hidden-layer sizes while keeping the activation functions fixed. For each configuration, evaluate the learning dynamics (epochs to convergence, training and validation error curves, test accuracy). Additionally, examine the impact of the mini-batch size._

---

## How To Install

### 1. Clone repository

Open **command prompt** and enter:

    git clone <repository_url>

### 2. Create Virtual Environment

Navigate into the **directory** '_P12-PROJECT_' and enter:

    python -m venv venv

### 3. Activate Virtual Environment

Into the same directory enter:

**Windows**

    .\/venv/Scripts/activate

**MacOS/Linux**

    source venv/bin/activate

### 3. Install dependecies

Into the virtual environment enter:

    python -m pip install -r requirements.txt

### 4. Start Training

To start training enter:

    python main.py

### 5. Check Predict [optional]

Change the number of the image_path into "predict.py" (The images are in the dir "images") and enter:

    python predict.py

**Note:** The `predict.py` script was created quickly to test whether the model can predict images outside the MNIST dataset. The test images were drawn using Paint. Feel free to modify the script or use new images to improve the results.
