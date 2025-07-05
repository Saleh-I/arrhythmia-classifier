# LSTM-Based Irregular Heartbeat Classification (INCART Dataset)

This project presents an LSTM-based deep learning model trained to classify irregular heartbeats from the INCART 12-lead ECG dataset.

## Motivation

Inspired by Prof. Martijn Boussé’s work on Kronecker Product Equations for ECG classification, this project explores a neural sequence model to address the same classification challenge using deep learning.

## Dataset

- **Source**: [INCART 12-lead Arrhythmia Database (PhysioNet)](https://physionet.org/content/incartdb/1.0.0/)
- **Structure**: 75 ECG recordings from 32 subjects (12 leads, 30 minutes each)
- **Classes**: 4 heartbeat types (grouped)
- **Split**: 80% training / 20% validation

## Data Preprocessing

The preprocessing script extracts 1-second windows (256 samples) around R-peaks using WFDB annotations.

### Highlights
- Sampling frequency: 257 Hz
- Extracts ECG segments of shape `(256, 12)` centered on each annotated beat
- Z-score normalization per window and per lead
- Removes non-beat symbols (e.g., '+') and incomplete edge samples

### Label Mapping
Heartbeat annotations were grouped into 4 categories:

| Group         | Original Labels             | One-Hot Vector         |
|---------------|-----------------------------|------------------------|
| Normal        | `'N', 'R', 'L', 'n', 'B'`    | `[1, 0, 0, 0]`         |
| SVEB (Supra.) | `'A', 'S', 'j'`              | `[0, 1, 0, 0]`         |
| VEB (Ventr.)  | `'V'`                        | `[0, 0, 1, 0]`         |
| Other         | All remaining labels         | `[0, 0, 0, 1]`         |

## Method

- **Model**: LSTM neural network (implemented in PyTorch)
- **Input**: `(256, 12)` ECG windows
- **Target**: One-hot encoded heartbeat class
- **Training**: CrossEntropyLoss (implicitly from one-hot format), Adam optimizer
- **Evaluation**: Confusion Matrix, Precision, Recall, F1-Score

## Results

### Confusion Matrix

|                  | Pred: Normal | Pred: SVEB | Pred: VEB | Pred: Other |
| ---------------- | ------------ | ---------- | --------- | ----------- |
| **True: Normal** | 30,632       | 23         | 12        | 0           |
| **True: SVEB**   | 59           | 346        | 0         | 0           |
| **True: VEB**    | 24           | 0          | 4018      | 0           |
| **True: Other**  | 14           | 0          | 36        | 0           |

### Classification Report
```
               precision    recall  f1-score   support

      Normal       1.00      1.00      1.00     30667
        SVEB       0.94      0.85      0.89       405
         VEB       0.99      0.99      0.99      4042
       Other       0.00      0.00      0.00        50

    Accuracy                           1.00     35164
   Macro Avg       0.73      0.71      0.72     35164
Weighted Avg       0.99      1.00      0.99     35164
```


| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0     | 1.00      | 1.00   | 1.00     |
| 1     | 0.94      | 0.85   | 0.89     |
| 2     | 0.99      | 0.99   | 0.99     |
| 3     | 0.00      | 0.00   | 0.00     |

**Overall Accuracy**: **~99.9%**

⚠️ **Note**: Class 3 (Other) performance is poor due to strong class imbalance, few samples are available for this category.

## Future Work

- Apply class weighting or oversampling to balance training
- Compare performance with tensor-based methods like Kronecker Product Equations (KPE)
- Investigate hybrid models that combine deep learning with structured signal embeddings

## References

- Boussé, M. et al. *Irregular Heartbeat Classification Using Kronecker Product Equations*, KU Leuven, 2017.  
  🔗 [https://ftp.esat.kuleuven.be/pub/pub/SISTA/mbousse/reports/mbousse2017ihc.pdf](https://ftp.esat.kuleuven.be/pub/pub/SISTA/mbousse/reports/mbousse2017ihc.pdf)


