# Depression Detection System

## 📌 Overview

The **AI Depression Detection System** is a multimodal machine learning application designed to analyze different forms of user-generated social media data to identify patterns that may be associated with depressive tendencies.

The system combines **text, facial expressions, and emoji-based emotional information** to provide a more comprehensive analysis than relying on a single data modality.

> **Disclaimer:** This project is developed for academic and research purposes. It is not a medical diagnostic tool and should not be used as a substitute for professional mental-health assessment.

---

## 🎯 Objectives

The main objectives of this project are:

- Analyze textual content for sentiment and emotional patterns.
- Detect facial emotions from images.
- Analyze emoji sequences to identify emotional patterns.
- Combine multiple modalities using multimodal feature fusion.
- Develop an integrated application for depression-risk analysis.
- Demonstrate the practical application of AI and deep learning in mental-health research.

---

## 🧠 System Architecture

The system processes three primary modalities:

### 1. Text Analysis

Textual data is processed using **DistilBERT** to extract contextual sentiment and emotional features.

**Technology:** DistilBERT / NLP

### 2. Facial Emotion Recognition

Images are analyzed using a **Convolutional Neural Network (CNN)** to identify facial emotion patterns.

**Technology:** CNN / Computer Vision

### 3. Emoji Emotion Analysis

Emoji sequences are analyzed using a **Bidirectional LSTM (Bi-LSTM)** to capture emotional patterns and sequence dependencies.

**Technology:** Bi-LSTM / Deep Learning

### 4. Multimodal Fusion

The features extracted from text, images, and emojis are combined using a multimodal fusion approach to produce an integrated prediction.

```text
             User Data
                 │
        ┌────────┼────────┐
        │        │        │
      Text     Image     Emoji
        │        │        │
        ▼        ▼        ▼
    DistilBERT  CNN    Bi-LSTM
        │        │        │
        └────────┼────────┘
                 │
        Multimodal Feature
             Fusion
                 │
                 ▼
       Depression Analysis