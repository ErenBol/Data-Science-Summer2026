# ML Foundations to Neural NLP — Notebook Sequence

This folder contains a connected six-notebook curriculum designed for a learner who already knows basic statistics, supervised machine learning, and classification.

## Recommended order

1. `01_learning_theory_polynomial_function_spaces.ipynb`
   - computational/statistical learning theory
   - PAC intuition and sample complexity
   - generalization
   - polynomial regression and complete polynomial bases
   - function spaces and regularization

2. `02_clustering.ipynb`
   - K-means from scratch
   - initialization, scaling, diagnostics, and failure modes
   - DBSCAN and hierarchical clustering

3. `03_perceptron_ann_numpy_pytorch_keras.ipynb`
   - perceptron
   - MLP and backpropagation from scratch with NumPy
   - PyTorch and Keras equivalents

4. `04_cnn_numpy_pytorch_keras.ipynb`
   - convolution and pooling from scratch
   - visual feature maps
   - PyTorch and Keras CNNs

5. `05_markov_bigram_hmm_pos.ipynb`
   - Markov chains
   - smoothed bigram language models
   - HMMs and Viterbi POS tagging

6. `06_rnn_lstm_numpy_pytorch_keras.ipynb`
   - recurrent hidden state and BPTT
   - vanishing/exploding gradients
   - LSTM gates
   - PyTorch and Keras sequence models

## Environment

Core notebooks use NumPy, matplotlib, scikit-learn, and PyTorch. TensorFlow/Keras sections are optional and skip gracefully when TensorFlow is unavailable.

Suggested installation:

```bash
pip install numpy matplotlib scikit-learn jupyter torch
```

Optional Keras sections:

```bash
pip install tensorflow
```

## Study method

For each notebook:

1. Read the explanation before running the code.
2. Predict the output or plot shape.
3. Run each cell and modify one assumption.
4. Complete at least two exercises.
5. Write a short summary from memory before moving on.

The final notebook deliberately ends with the conceptual bridge needed before transformer and BERT study, without teaching BERT itself.
