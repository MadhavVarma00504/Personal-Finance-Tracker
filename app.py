from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///finance.db'
db = SQLAlchemy(app)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(100), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(100), nullable=False)
    type = db.Column(db.String(100), nullable=False)  # Income or Expense

# Initialize the database
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    transactions = Transaction.query.all()
    return render_template('index.html', transactions=transactions)

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    date = request.form['date']
    amount = request.form['amount']
    category = request.form['category']
    type = request.form['type']
    new_transaction = Transaction(date=date, amount=amount, category=category, type=type)
    db.session.add(new_transaction)
    db.session.commit()
    return 'Transaction added!'

def preprocess_data():
    transactions = Transaction.query.all()
    df = pd.DataFrame([(t.date, t.amount, t.type) for t in transactions], columns=['Date', 'Amount', 'Type'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Amount'] = df['Amount'].astype(float)
    df['Type'] = df['Type'].apply(lambda x: 1 if x == 'Income' else 0)
    
    # Add a feature for time series prediction (e.g., cumulative balance over time)
    df['Balance'] = df['Amount'].cumsum()
    return df

class FinanceDataset(Dataset):
    def __init__(self, data, sequence_length=60):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        X = self.data[index:index+self.sequence_length]
        y = self.data[index+self.sequence_length]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

def build_lstm_model(df):
    # Prepare the data for time series prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Balance'].values.reshape(-1, 1))

    # Create dataset and dataloader
    dataset = FinanceDataset(scaled_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Build the LSTM model
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()
    epochs = 10
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.unsqueeze(-1)  # Reshape for LSTM input
            output = model(X_batch)
            loss = criterion(output, y_batch.unsqueeze(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, scaler

def predict_future_balance(model, scaler, df):
    # Prepare the last 60 days of data for prediction
    model.eval()
    last_60_days = df['Balance'].values[-60:]
    last_60_scaled = scaler.transform(last_60_days.reshape(-1, 1))

    X_test = torch.tensor(last_60_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # Reshape for LSTM input
    with torch.no_grad():
        predicted_balance = model(X_test)
    
    predicted_balance = scaler.inverse_transform(predicted_balance.numpy())
    return predicted_balance[0][0]

@app.route('/train_model')
def train_model():
    df = preprocess_data()
    if len(df) < 60:
        return "Not enough data to train the model."
    
    model, scaler = build_lstm_model(df)
    predicted_balance = predict_future_balance(model, scaler, df)
    return f'Predicted future balance: {predicted_balance}'

if __name__ == "__main__":
    app.run(debug=True)
