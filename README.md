# Fraud Detection Terminal

An interactive web application built to detect fraudulent credit card transactions in real-time using a high-performance machine learning model.

## Overview

This project tackles the challenge of fraud detection on a highly imbalanced dataset where fraudulent transactions account for less than 0.2% of the data. The core of the application is a **LightGBM** classifier, a powerful gradient boosting model known for its speed and accuracy.

**Key Features & Techniques:**
-   **Extreme Imbalance Handling:** Uses the **Random Under-sampling** technique to create a balanced training set, allowing the model to learn fraud patterns effectively.
-   **Robust Workflow:** Implements a correct ML pipeline that splits data *before* preprocessing to prevent data leakage.
-   **High-Performance Model:** Employs a LightGBM model, achieving a strong **AUC-PR score** for reliable fraud identification.
-   **Modern UI:** Features a custom-themed Gradio interface with a "security terminal" look and feel for an enhanced user experience.

## Live Demo

You can try the live application here: **[Your Hugging Face Space URL]**

## How to Use

The model requires 30 specific features from a transaction to make a prediction. To use the app, provide these 30 numerical values as a single, comma-separated string in the input box.

The required order is: **V1, V2, ..., V28, Time, Amount**

**Example (Fraudulent Transaction):**
