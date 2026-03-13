📈 Stock Price Prediction using LSTM Neural Network
A deep learning project that predicts stock prices using a multi-layered Long Short-Term Memory (LSTM) neural network, trained on 10 years of real historical stock data fetched automatically from Yahoo Finance.

🧠 How the Model Works
The model uses a stacked LSTM architecture to learn patterns from historical closing prices and predict future stock prices. LSTM networks are particularly well suited for this kind of task because they are designed to work with data that changes over time — they can remember important patterns from the past and use them to make smarter predictions about the future.
The network is built with four LSTM layers stacked on top of each other, with 50, 60, 80, and 120 units respectively. After each LSTM layer, a Dropout layer is added with increasing rates of 20%, 30%, 40%, and 50% to prevent the model from overfitting, which is when a model memorizes the training data too closely and fails to generalize to new data. A final Dense layer with a single output unit produces the predicted stock price. The model is compiled using the Adam optimizer and Mean Squared Error as the loss function, and trained over 50 epochs using ReLU activation.

📊 Features
The model accepts any valid stock ticker symbol such as AAPL, TSLA, or RELIANCE.NS and automatically downloads the last 10 years of daily historical data in real time. It also fetches the current live price of the stock so you always have an up to date reference point. Throughout the notebook, several charts are generated — the raw closing price over time, the 100-day and 200-day moving averages overlaid on the closing price, and finally a comparison of the actual prices versus the model's predicted prices on the test set.

🗂️ Dataset & Preprocessing
Data is sourced from Yahoo Finance via the yfinance library, covering the last 10 years at a daily interval. Only the closing price is used as the input feature. Before training, the closing prices are scaled down to a range between 0 and 1 using MinMaxScaler, which helps the model learn more efficiently. The data is then split with the first 80% used for training and the remaining 20% reserved for testing. The model is given a window of 100 consecutive days at a time and trained to predict the price on the 101st day. The last 100 days of training data are also prepended to the test set to maintain proper sequence continuity during evaluation.

🛠️ Tech Stack
This project is built with Python 3.13 using TensorFlow and Keras for the deep learning model, NumPy and Pandas for data handling, Matplotlib for visualizations, yfinance for fetching real-time stock data, and Scikit-learn for data normalization.

🚀 Getting Started
Clone the repository and install the required dependencies using the commands below.
bashgit clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
pip install numpy pandas matplotlib yfinance scikit-learn tensorflow
Then open the notebook and run all cells. When prompted, enter any stock ticker symbol to begin.
bashjupyter notebook stockprice.ipynb
The trained model will be saved automatically at the end of the notebook as Stock Price Prediction Model.keras.

📁 Project Structure
stock-price-prediction/
│
├── stockprice.ipynb                   # Main Jupyter Notebook
├── Stock Price Prediction Model.keras # Saved trained model
└── README.md                          # Project documentation

⚠️ Disclaimer
This project is built purely for educational purposes. The predictions made by this model should not be used for real financial decisions or investment advice. Stock markets are influenced by many unpredictable factors that cannot be captured by historical price data alone.
