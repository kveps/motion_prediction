# Motion Prediction

This project implements a motion prediction model for actors (vehicles, pedestrians, etc.) using the Waymo Open Motion Dataset. It aims to predict future trajectories of these actors based on their historical movements. This implementation includes both a LSTM and Transformer-based models.

## Project Structure

The repository is organized as follows:

-   `models/`: Contains the implementation of the prediction models.
    -   `lstm/`: Long Short-Term Memory (LSTM) model.
    -   `transformer/`: Transformer-based model.
    -   `loss/`: Custom loss functions.
    -   `trained_weights/`: Directory to save and load trained model weights.
-   `scripts/`: Contains scripts for training, testing, and visualization.
    -   `lstm_train.py`: Script to train the LSTM model.
    -   `transformer_train.py`: Script to train the Transformer model.
    -   `lstm_test.py`: Script to test the LSTM model.
    -   `animate_tf_example.py`: Script to visualize a data example and create an animation.
    -   `mount_data.sh`: Shell script to mount the Waymo dataset from Google Cloud Storage.
-   `utils/`: Contains utility functions for data processing and visualization.
    -   `data/`: Helper functions for handling the Waymo dataset.
    -   `viz/`: Functions for visualizing scenarios and creating animations.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/motion-prediction.git
    cd motion-prediction
    ```

2.  **Install dependencies:**

    This project requires Python 3 and the following packages. You can install them using pip:

    ```bash
    pip install tensorflow ipython
    ```

    You will also need to install `gcsfuse` to mount the dataset. Follow the instructions [here](https://cloud.google.com/storage/docs/gcsfuse-quickstart-mount-bucket) if it's not yet installed.

3.  **Download the data:**

    The Waymo Open Motion Dataset is required. This project provides a script to mount the dataset from a Google Cloud Storage bucket.

    First, authenticate with Google Cloud:

    ```bash
    gcloud auth application-default login
    ```

    Then, run the script to mount the data:

    ```bash
    ./scripts/mount_data.sh
    ```

    This will create a `data/` directory in the project root and mount the dataset there.

## Usage

### Visualize the data

To understand the dataset, you can visualize a scenario. The `animate_tf_example.py` script creates an HTML animation of a given data example.

```bash
python scripts/animate_tf_example.py
```

You will be prompted to enter the path to a `.tfrecord` file. If you leave it empty, it will use a default path. The output will be saved as `tmp/data.html`, which you can open in a web browser.

### Train a model

You can train either the LSTM or the Transformer model.

**Train the LSTM model:**

```bash
python scripts/lstm_train.py
```

**Train the Transformer model:**

```bash
python scripts/transformer_train.py
```

Trained model weights will be saved in the `models/trained_weights/` directory.

### Test a model

To test a trained model, you can use the provided test scripts.

**Test the LSTM model:**

```bash
python scripts/lstm_test.py
```

## Models

This project provides two deep learning models for motion prediction:

-   **LSTM (Long Short-Term Memory):** A recurrent neural network (RNN) architecture that is well-suited for sequence prediction tasks.
-   **Transformer:** A model architecture that relies on self-attention mechanisms to process sequential data.
