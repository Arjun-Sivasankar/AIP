# Access Interval Prediction (AIP)

Neural predictors of memory-access intervals in hardware accelerators, trained on execution traces. Given the recent history of an access stream, the model predicts the interval until the next access, which is the quantity a hardware predictor needs in order to act before the access happens rather than after it.

This is research code from my time at the **Vodafone Chair Mobile Communication Systems, TU Dresden**, working on the same problem as the IEEE ACDSA 2024 paper *Enlarging the Time Budget for Neural Network Based Predictors for Access Interval Prediction* (Friedrich, Sivasankar, Matúš, Fettweis).

## What is in here

| File | What it does |
|---|---|
| `training_preprocessing.py` | Loads trace files and turns addresses into hashed feature windows, with configurable hash shift, modulo and feature size |
| `training_helper.py` | Shared training utilities |
| `training_optuna.py` | Keras `Conv1D` classifier with TPE hyperparameter search over architecture and optimiser settings, scored with balanced accuracy and a confusion matrix |
| `training_quasi_online.py` | Simulates online training: reads the trace in chunks, trains in small batches and logs accuracy at fixed intervals |
| `training_test_offline.py`, `training_top.py` | Offline evaluation variants |
| `Generate_plot.py` | Plotly figures for the accuracy logs, with spline smoothing and a KDE |

The network is small on purpose: convolution and pooling layers, no large backbone.

## Running it

Each script is driven by constants at the top of the file rather than by a command line, and they read a trace file from the working directory.

```bash
pip install tensorflow keras optuna scikit-learn pandas numpy scipy plotly
python training_optuna.py
```

`Generate_plot.py` reads `eval_acc3.txt`, so the training script that writes that log has to run first.

## Status

Research code, published as it was used. It is not packaged as a library and there is no test suite, so treat it as a record of the experiments.
