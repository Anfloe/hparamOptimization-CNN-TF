# CIFAR-10 CNN Hyperparameter Search (Sequential vs Functional)

This repo contains a Jupyter notebook that **experiments with hyperparameter tuning** for two compact CNNs on **CIFAR-10**. It compares a **Sequential** baseline (Model A) and a **Functional** two-branch network (Model B), performs a **manual random search** over key hyperparameters, then **retrains the best configurations** by tweaking the training recipe. The goal is to study how random search affects results, how the two architectures behave, and how **overfitting vs. speed** trade-offs show up in practice.

**What it investigates**
- Accuracy across different hyperparameters  
- Generalization  
- Training efficiency

**A notable finding**
Fewer parameters did **not** always mean faster training. Runtime is dominated by **convolutional FLOPs** (operations per forward pass), which depend on spatial size and kernel work—not just parameter count. A model that performs **more convs before pooling** can be **slower per epoch** even with fewer parameters.


---

## This notebook explores:

1. **Manual random search** over:
   - optimizer, learning rate, initializer, activation, dropout, filters, batch size
2. **EarlyStopping + ReduceLROnPlateau + Checkpointing** to train the top configs efficiently
3. **Overfitting analysis** via the train–val gap
4. **Timing** (per-epoch and total) for each trial and final runs
5. Side-by-side comparison of accuracy vs. speed for the two architectures

---

## Models

- **Model A (Sequential):** three Conv blocks with BatchNorm/Dropout and pooling; Dense(10, softmax)
- **Model B (Functional):** two parallel Conv paths (3×3 and 5×5), each with BN/Dropout/Pool, **concat → Dense(10, softmax)**

> Losses are **multi-class** (loss is`categorical_crossentropy`).

---

## Dataset

- **CIFAR-10** (50k train / 10k test, 32×32×3)
- Notebook includes simple image scaling; labels can be one-hot or integer (choose matching loss).

---

## Results (final runs)

| Metric | **Model A (Sequential)** | **Model B (Functional)** |
|---|---:|---:|
| **Best val acc** | **0.7218** | 0.5973 |
| **Test acc** | **0.7199** | 0.5944 |
| **Train–Val gap** | 0.1119 | **0.0865** |
| **Val–Test gap** | **0.0019** | 0.0029 |
| Epochs (stopped) | ~37 | ~55 |
| ~Seconds / epoch | ~5.94 s | **~2.58 s** |
| Total time | ~220 s | **~142 s** |

**Takeaways**
- **Accuracy leader:** Model A (~0.72).  
- **Efficiency leader:** Model B (~2.3× faster/epoch) with slightly smaller train–val gap but lower ceiling (~0.60).  
- With BatchNorm, **light/no dropout (0–0.2)** and **moderate learning rates** worked best.  
- Tuning **optimizer/LR** mattered more than widening (32–64 filters were enough).

---

## How to run

1. Install dependencies
   ```bash
   pip install -r requirements.txt
   # For model diagrams (optional): install Graphviz system package if needed

2. Open notebook in Jupyter or Google Colab nd run cells sequentially.
3. GPU recommended. Training times are indicative and related to hardware.

## Editing the experiment
- **Search space:** tweak and lists (optimizers, learning rates, etc.)
- **Callbacks:** adjust EarlyStopping patience and LR reduction patience for longer/shorter runs.
- **Timing:** the notebook includes a small callback to log per-epoch and total time.

## Next steps (no architecture changes)
- Light on-GPU augmentation (RandomFlip/Translation/Contrast)
- Label smoothing (~0.05)
- Mixed precision on GPU for throughput



