# CIFAR-10 CNN Hyperparameter Search (Modernized)

This notebook explores **hyperparameter tuning** for CNNs on **CIFAR-10**, comparing a simple **Sequential** model (Model A) and a **Functional** model with two branches (Model B).  
The search is done with a **custom random-search loop**, and training uses a robust callback recipe (EarlyStopping, ReduceLROnPlateau, checkpointing).

---

## Contents

- **Model A (Sequential API)**  
  Three conv blocks with BatchNorm/Dropout and pooling.  
  - Tuned via custom random search (optimizer, LR, init, activation, dropout, filters, batch size).  
  - Best recent configs reached **~0.58–0.60 validation accuracy** with **Adam (1e-3), He init, ReLU, light/no dropout, 32–64 filters**.

- **Model B (Functional API)**  
  Two parallel conv paths (3×3 and 5×5) merged before classification.  
  - Tuned with the same random-search procedure.  
  - Best recent configs reached **~0.48–0.50 validation accuracy** (often faster per epoch due to fewer FLOPs despite similar parameter counts).

---

## Key Takeaways

- Moderate learning rates (≈ 1e-3 to 1e-2) and well-chosen optimizers (Adam/RMSprop; SGD with sensible LR) mattered more than doubling filters.  
- **Regularization:** With BatchNorm, light or no dropout (0.0–0.2) was most effective.  
- **Speed vs params:** Model B often trains **faster per epoch** even with similar/more params because it pools earlier → fewer conv FLOPs at high resolution.

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # If you want model diagrams:
   # (Linux) sudo apt-get install graphviz
   # (macOS) brew install graphviz
