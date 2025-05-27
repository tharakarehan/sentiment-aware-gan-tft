# Sentiment-Aware TFT-WGAN for Crypto Market Bottom Prediction

This repository implements a deep learning framework to predict market bottoms in cryptocurrency trading using multi-modal inputs and generative adversarial learning.

## 🚀 Overview

Cryptocurrency markets are highly volatile, with sudden trend reversals that challenge traditional forecasting methods. This project proposes a **Sentiment-Aware Temporal Fusion Transformer with Wasserstein GAN (TFT-WGAN)** to forecast price paths and detect market bottoms. A downstream **XGBoost classifier** uses hidden states from the GAN, combined with real-time sentiment and trend data, to estimate the probability that the next bar is a bottom.

## 🧠 Architecture

- **Generator**: Temporal Fusion Transformer (TFT)
- **Discriminator**: 1D CNN
- **Classifier**: XGBoost using generator latent states + sentiment/trend features

## 📊 Data Sources

- OHLCV data (Binance/Yahoo Finance)
- Sentiment (Twitter/Reddit via FinBERT/CryptoBERT)
- Google Trends (via PyTrends)
- Technical Indicators (SMA, EMA, RSI, MACD, etc.)

## 🧪 Evaluation Metrics

- Forecasting: RMSE, MAE
- Classification: AUROC, F1, Precision, Recall
- Strategy: Simulated returns under bottom-alert conditions

## 📦 Dependencies

- Python 3.9+
- TensorFlow / PyTorch
- XGBoost
- Scikit-learn
- Huggingface Transformers
- PyTrends
- Pandas, NumPy

## 📁 Project Structure

```text
project-root/
├── data/               # Scripts & storage for multimodal time-series data
├── models/             # TFT-WGAN and classifier modules
├── utils/              # Preprocessing, feature generation, evaluation
├── notebooks/          # Prototype and experiment notebooks
├── config/             # Hyperparameter and architecture configs
└── README.md           # Project overview and setup
```

## 🛠 Development Status

- ✅ Literature + Architecture Finalized
- 🛠 Model Training & Evaluation in progress
- 📅 MVP Target: **July 2025**

## 📄 License

MIT License © 2025 Tharaka Rehan
