
# CIFAR-10 CNN Hyperparameter Search

This project explores **hyperparameter tuning** for convolutional neural networks (CNNs) on the **CIFAR-10 dataset**.  
The notebook compares two model architectures (Sequential vs. Functional API) and two search strategies (scikit-learn’s `RandomizedSearchCV` vs. a custom random-search loop).

---

## Contents
- **Model A (Sequential API):**  
  A baseline CNN with three convolutional blocks.  
  - Hyperparameter search with `RandomizedSearchCV`  
  - Custom random-search loop for more control  
  - Reached modest performance (~30% accuracy in CV; ~70% in custom search)

- **Model B (Functional API):**  
  A more flexible CNN with two parallel convolutional paths (3×3 and 5×5 kernels) merged before classification.  
  - Same random-search procedure applied  
  - Consistently higher accuracy than the Sequential model  
  - Best configuration reached **~94% validation accuracy**

- **Final evaluation:**  
  The best functional model was trained with:  
  - Optimizer: **Adadelta (lr=0.001)**  
  - Init: **he_normal**  
  - Activation: **ReLU**  
  - Dropout: **0.2**  
  - Filters: **64**  
  - Batch size: **64**

  Achieved **~90.6% test accuracy** with stable validation curves.

---

## Key Takeaways
- The **Functional API model** outperforms the Sequential CNN, showing the benefit of multi-branch architectures.  
- Random search can surface strong hyperparameter sets, but visualizing results makes comparison clearer.  
- Early stopping helps prevent overfitting and speeds up training.  
- Note: The notebook uses `binary_crossentropy` for continuity. For multi-class problems like CIFAR-10, the standard choice is `categorical_crossentropy` or `sparse_categorical_crossentropy`.

---

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

