Sentiment-Aware TFT-WGAN for Crypto Market Bottom Prediction
This repository implements a two-stage deep learning framework for detecting cryptocurrency market bottoms using multi-modal data and generative adversarial learning.

🚀 Overview
Cryptocurrency markets are volatile and prone to abrupt reversals. This project builds a Sentiment-Aware Temporal Fusion Transformer in a WGAN-GP setup to forecast price paths and detect market bottoms in real time. A downstream XGBoost classifier uses latent features and sentiment inputs to output bottom probabilities.

🧠 Core Architecture
TFT-WGAN-GP

Generator: Temporal Fusion Transformer (multi-horizon forecasting)

Discriminator: CNN (sequence realism enforcement)

Bottom Classifier

Input: Latent state from TFT + recent sentiment & trends

Model: XGBoost

📊 Data Sources
Binance/Yahoo Finance (OHLCV @ 5-min intervals)

Twitter/Reddit sentiment (via FinBERT/CryptoBERT)

Google Trends (via PyTrends)

Engineered technical indicators (SMA, RSI, MACD, etc.)

🧪 Evaluation Metrics
Forecasting: RMSE, MAE

Classification: AUROC, Precision, Recall, F1

Trading Simulations: Risk-adjusted returns

📦 Key Dependencies
Python 3.9+, TensorFlow, PyTorch

XGBoost, Scikit-learn

Huggingface Transformers, PyTrends

📁 Project Structure
graphql
Copy
Edit
├── data/               # Scripts & storage for multimodal data
├── models/             # TFT-WGAN and XGBoost implementations
├── utils/              # Preprocessing, feature engineering
├── notebooks/          # PoC and experiment runs
├── config/             # Model and training configs
└── README.md
📍 Status
🚧 Work in progress — final model & experiments underway.
📅 MVP release planned: July 2025

📄 License
MIT License © 2025 Tharaka Rehan
