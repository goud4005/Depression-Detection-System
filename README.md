# 🧠 Deepression — Multimodal Depression Indicator Analysis System

> An experimental multimodal AI system that analyzes text, facial expressions, and emojis to identify patterns associated with depression-related indicators.

---

## 📌 Overview

AIDeepression is a Flask-based machine learning application designed to explore multimodal approaches for analyzing depression-related signals from user-provided content.

Instead of relying on a single input type, the system combines multiple modalities:

- 📝 Text analysis
- 🖼️ Facial/image analysis
- 😊 Emoji analysis
- 🔀 Multimodal score fusion
- 🔍 SHAP-based text explainability
- 🔥 Grad-CAM-based image visualization
- 📊 User history and dashboard analytics

The project is intended for **academic, research, and educational purposes**.

It should not be considered a medical diagnostic system.

---

## ✨ Features

### 📝 Text Analysis

The system processes user-provided text and uses a transformer-based model to estimate depression-related signals.

The application:

- Cleans the input text
- Removes emojis for text-only analysis
- Generates a prediction score
- Classifies the result as Normal or Depression
- Generates SHAP-based explanations

---

### 🖼️ Facial/Image Analysis

The application can analyze an uploaded image using trained deep-learning models.

The image pipeline provides:

- Image preprocessing
- Multiple model predictions
- Depression-related probability estimation
- Grad-CAM++ visualization

Grad-CAM++ is used to provide a visual explanation of regions that contributed to the model prediction.

---

### 😊 Emoji Analysis

Emojis are analyzed separately using a trained neural network model.

The system extracts emoji-related patterns and generates an additional prediction score.

---

## 🔀 Multimodal Fusion

The final prediction combines the available modalities.

Current fusion weights:

| Modality | Weight |
|----------|-------:|
| Text | 60% |
| Emoji | 30% |
| Image | 10% |

If only some modalities are available, the system normalizes the weights across the available inputs.

### Fusion Formula

```text
Fusion Score =
Σ(modality score × modality weight)
-----------------------------------
        Σ(modality weights)