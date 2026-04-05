# comp3361ass2
# NER-Bench: Neural Architectures for Named Entity Recognition

This repository implements and evaluates multiple neural architectures to solve the **Named Entity Recognition (NER)** task on the OntoNotes 5.0 dataset.

## 🏗 Supported Architectures

| Mode | Architecture | Key Features |
| :--- | :--- | :--- |
| `lstm` | **BiLSTM** | Bidirectional LSTM + Linear Layer + Optimized Class Weighting |
| `trans` | **Transformer** | 4-layer Encoder + Sinusoidal Positional Encoding + Label Smoothing |
| `bert` | **DistilBERT** | Pre-trained Transformer Fine-tuning with Subword Alignment |

---

## 🚀 Getting Started
# NER Project Report
View the full experimental report here: [📄 Download/View PDF Report](./3036128157.pdf)
### 1. Model Configuration
Before running the pipeline, specify the architecture you wish to train by modifying the `mode` variable in `train.py`:

```python
# train.py (Line ~14)
mode = "lstm"  # Options: "lstm", "trans", "bert"



AI Usage Declaration

Tool Used: Gemini

Nature of Assistance:

Conceptual Understanding: Clarified the mathematical principles of Sinusoidal Positional Encoding and its role in Transformer-based NER.

Code Verification: Consulted for best practices in PyTorch implementation, specifically regarding register_buffer and embedding scaling.

Data Synthesis: Assisted in structuring and refining the Experimental Report and Analysis based on the results obtained from my local grid search.

Documentation: Provided suggestions for README formatting.




