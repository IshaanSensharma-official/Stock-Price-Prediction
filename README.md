📈 Stock Price Prediction using LSTM Neural Network

A deep learning experiment in time‑series forecasting — enter any ticker, watch the model learn 10 years of market data.

🧠 How the Model Works 
The model uses a stacked LSTM architecture to learn patterns from historical closing prices and predict future stock prices. LSTM networks are particularly well suited for this kind of task because they are designed to work with data that changes over time — they can remember important patterns from the past and use them to make smarter predictions about the future. The network is built with four LSTM layers stacked on top of each other, with 50, 60, 80, and 120 units respectively. After each LSTM layer, a Dropout layer is added with increasing rates of 20%, 30%, 40%, and 50% to prevent the model from overfitting, which is when a model memorizes the training data too closely and fails to generalize to new data. A final Dense layer with a single output unit produces the predicted stock price. The model is compiled using the Adam optimizer and Mean Squared Error as the loss function, and trained over 50 epochs using ReLU activation.

🔬 Architecture Deep Dive
The network uses four stacked LSTM layers with progressively increasing capacity, followed by a single Dense output unit that predicts the next closing price.

Why progressive dropout?
Each layer's dropout rate increases from 20% → 50%, forcing the network to learn progressively more abstract and robust representations. This combats co-adaptation between neurons and significantly improves generalization on unseen market data.
Key hyperparameters:

Activation: ReLU — stabilises gradient flow through deep layers
Loss function: Mean Squared Error (MSE)
Optimizer: Adam
Epochs: 10


📊 Features
🔍 Any stock ticker — Works with global symbols (AAPL, TSLA, RELIANCE.NS, ^NSEI, and more)
🌐 Live data fetch — Automatically downloads the last 10 years of daily closing prices from Yahoo Finance
💹 Live current price — Fetches today's price for real-time context alongside predictions
📈 Auto-generated visualisations — Three plots are produced on every run:

Raw closing price history
Closing price overlaid with 100-day & 200-day moving averages
Actual vs. predicted prices on the held-out test set

🗂️ Dataset & Preprocessing
Source: Yahoo Finance via yfinance
Interval: Daily, last 10 years
Feature: Univariate — closing price only
Pipeline:

Scaling — MinMaxScaler normalises prices to [0, 1] for faster, more stable convergence
Train / Test split — First 80% of data is used for training; last 20% for evaluation
Sequence construction — A sliding window of 100 days is used to predict day 101
Continuity preservation — The last 100 rows of training data are prepended to the test set to ensure uninterrupted sequence context at the boundary


🛠️ Tech Stack
Language: Python 3.13
Deep Learning: TensorFlow / Keras
Data Handling: NumPy, Pandas
Visualisation: Matplotlib
Data Source: yfinance
Preprocessing: Scikit-learn (MinMaxScaler)


🚀 Getting Started
Clone the repository
bashgit clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
Install dependencies
bashpip install numpy pandas matplotlib yfinance scikit-learn tensorflow
Launch the notebook
bashjupyter notebook stockprice.ipynb
When prompted, enter any valid stock ticker (e.g. AAPL, TSLA, RELIANCE.NS) and the model will handle the rest. The trained model is automatically saved to disk upon completion.
Saved model output:
📁 Stock Price Prediction Model.keras

📁 Project Structure
stock-price-prediction/
│
├── stockprice.ipynb                     # Main notebook — run this
├── Stock Price Prediction Model.keras   # Saved model (generated after training)
└── README.md                            # Project documentation

⚠️ Disclaimer
This project is purely educational. Stock markets are governed by countless unpredictable variables macroeconomic shifts, geopolitical events, investor sentiment that historical price data alone cannot capture.


🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to open a pull request or file an issue on the repository.

Built with curiosity and coffee. ☕
