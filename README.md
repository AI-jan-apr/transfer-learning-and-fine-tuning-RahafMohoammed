# Transfer Learning with EfficientNetB0 — Food11 Dataset

## Experiment Summary

This assignment applies Transfer Learning using EfficientNetB0 pretrained on ImageNet to classify food images across 11 categories from the Food11 dataset.

Two experiments were conducted:

**Experiment 1 — Feature Extraction:** All base model layers were frozen. Only the classification head (GlobalAveragePooling2D + Dropout + Dense) was trained. This treats EfficientNet as a fixed feature extractor.

**Experiment 2 — Fine-tuning:** The top 20 layers of the base model were unfrozen (with BatchNormalization layers kept frozen) and retrained with a lower learning rate (1e-5 vs 1e-3). This allows the model to adapt its higher-level features to the food domain.

---

## Dataset

- **Source:** Food11 Image Dataset (Kaggle — trolukovich)
- **Classes (11):** Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles-Pasta, Rice, Seafood, Soup, Vegetable-Fruit
- **Split:** 80% training / 20% validation
- **Input size:** 224 x 224 x 3

---

## Model Architecture

```
Input (224, 224, 3)
        ↓
EfficientNetB0 (pretrained on ImageNet, include_top=False)
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.3)
        ↓
Dense (11, activation=softmax)
```

---

## Experiment 1 — Feature Extraction

| Setting | Value |
|---|---|
| Base model | Frozen (trainable=False) |
| Trainable params | 14,091 (head only) |
| Optimizer | Adam (lr=1e-3) |
| Loss | sparse_categorical_crossentropy |
| Epochs | 20 (EarlyStopping patience=5) |
| Batch size | 16 |

**Callbacks used:**
- EarlyStopping (patience=5, restore_best_weights=True)
- ReduceLROnPlateau (patience=3, factor=0.5)

**Observation:** Feature extraction converges quickly since only the small classification head is updated. The pretrained EfficientNet features from ImageNet transfer well to food classification because both domains share low-level visual patterns (edges, textures, colors).

---

## Experiment 2 — Fine-tuning

| Setting | Value |
|---|---|
| Unfrozen layers | Last 20 layers of base model |
| BatchNorm layers | Kept frozen (training=False) |
| Optimizer | Adam (lr=1e-5) |
| Loss | sparse_categorical_crossentropy |
| Epochs | 20 (EarlyStopping patience=5) |

**Why BatchNorm stays frozen:** Unfreezing BatchNormalization layers during fine-tuning causes the running mean and variance statistics to be overwritten by the small new dataset, which destroys the learned representations.

**Why smaller LR:** The pretrained weights are already good. A large LR would destroy them. A small LR (1e-5) nudges them gradually toward the new domain.

**Observation:** Fine-tuning improves performance over feature extraction because the top convolutional layers, which detect higher-level patterns (specific textures, shapes), get adapted to food-specific features rather than relying solely on general ImageNet features.

---

## Fine-tuning Techniques Explored

**Unfreeze only last N layers:** Instead of unfreezing the entire model, only the top 20 layers were unfrozen. Early layers detect general features (edges, gradients) that transfer well across domains. Later layers are more task-specific and benefit from fine-tuning.

**Gradual unfreezing:** An optional technique where layers are unfrozen progressively (e.g., last 10 → last 20 → last 40) across multiple training rounds. This gives the model time to stabilize at each stage before adapting deeper layers.

**Layer-wise learning rate decay:** A technique where earlier (deeper) layers use a smaller LR than later layers. This prevents destroying well-learned general features while allowing task-specific layers to adapt more aggressively.

---

## Observations

**Feature Extraction vs Fine-tuning:**
Fine-tuning consistently outperforms pure feature extraction when given enough data. The additional trainable parameters allow the model to adapt EfficientNet's higher-level representations to the specific visual patterns of food.

**Generalization:**
Both experiments used Dropout (0.3) and data augmentation (RandomFlip, RandomRotation) to reduce overfitting. EarlyStopping with restore_best_weights ensures the saved model is the best validation checkpoint.

**Overfitting risk:**
Fine-tuning with a small dataset or too high a learning rate risks overfitting. The low LR (1e-5) and frozen BatchNorm layers mitigate this.

**Convergence:**
Feature extraction converges faster (fewer parameters to optimize). Fine-tuning requires more epochs to stabilize but reaches a better final accuracy.

---

## Environment Note

Training was run on Google Colab CPU. Due to the absence of GPU, experiments were validated on a subset of the dataset (10%) to confirm correctness of the pipeline. Full training on GPU (T4) is expected to run in approximately 20-30 minutes per experiment and yield higher accuracy.

---

## Helpful References

- EfficientNet in Keras: https://keras.io/api/applications/efficientnet/
- Transfer Learning guide: https://keras.io/guides/transfer_learning/
- MLflow experiment tracking: https://www.mlflow.org/docs/latest/index.html
- Freeze/Unfreeze in Keras: https://keras.io/getting_started/faq/#how-can-i-freeze-layers-in-a-model
