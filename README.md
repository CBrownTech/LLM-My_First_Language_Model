# My First LSTM Language Model

This project implements a basic language model for next-word prediction using an LSTM (Long Short-Term Memory) network with PyTorch. The model is trained on text data extracted from a `.ndjson` file.

## Description

The script performs the following steps:
1.  **Data Loading**: Reads text data from a specified `.ndjson` file, targeting the `abstract` field of each JSON object.
2.  **Preprocessing**:
    *   Tokenizes the text using NLTK.
    *   Builds a vocabulary based on the most frequent words.
    *   Converts the text into sequences of token IDs.
    *   Creates input/target pairs for training the language model.
3.  **Model Definition**: Defines a simple language model using `nn.Embedding`, `nn.LSTM`, and `nn.Linear` layers.
4.  **Training**:
    *   Trains the model using the Adam optimizer and Cross-Entropy loss.
    *   Automatically detects and utilizes a CUDA-enabled GPU if available, falling back to the CPU otherwise.
    *   Prints the average loss for each epoch.

## Requirements

*   Python 3.x
*   PyTorch
*   NumPy
*   NLTK

## Installation

1.  Clone the repository or download the `First LLM Built.py` script.

2.  Install the required Python packages:
    ```bash
    pip install torch numpy nltk
    ```

3.  The script will automatically download the `punkt` tokenizer model from NLTK on its first run.

## Data

The script is configured to read data from a `.ndjson` file located at `C:\Users\doubl\Downloads\people_0.ndjson`. Each line in this file should be a JSON object containing a text field named `abstract`.

Example line in `.ndjson` file:
```json
{"id": "some_id", "abstract": "This is the text that will be used for training the model."}
