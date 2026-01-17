import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from quant_sandbox.common import set_seed
from quant_sandbox.macro_df import (
    BBKMCOIX,
    CORESTICKM159SFRBATL as CORE,
    FEDFUNDS,
    M2REAL,
    PAYEMS,
    UNRATE,
)


class FedFundsDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class FedRatePredictor(nn.Module):
    def __init__(self, input_dim):
        super(FedRatePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Prevent overfitting
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output is a single continuous value (The Rate)
        )

    def forward(self, x):
        return self.net(x)


def prepare_data():
    bbk = BBKMCOIX.with_columns([
        (pl.col("BBKMCOIX").pct_change(1) * 100).alias("BBKMCOIX % chg 1M"),
        #(pl.col("BBKMCOIX").pct_change(3) * 100).alias("BBKMCOIX % chg 3M"),
        #(pl.col("BBKMCOIX").pct_change(6) * 100).alias("BBKMCOIX % chg 6M"),
        #(pl.col("BBKMCOIX").pct_change(12) * 100).alias("BBKMCOIX % chg 12M"),
        #(pl.col("BBKMCOIX").pct_change(24) * 100).alias("BBKMCOIX % chg 24M"),
    ])

    m2real = M2REAL.with_columns([
        #(pl.col("M2REAL").pct_change(1) * 100).alias("M2REAL % chg 1M"),
        #(pl.col("M2REAL").pct_change(3) * 100).alias("M2REAL % chg 3M"),
        #(pl.col("M2REAL").pct_change(6) * 100).alias("M2REAL % chg 6M"),
        (pl.col("M2REAL").pct_change(12) * 100).alias("M2REAL % chg 12M"),
        #(pl.col("M2REAL").pct_change(24) * 100).alias("M2REAL % chg 24M"),
    ]).drop("M2REAL")

    payems = PAYEMS.with_columns([
        #(pl.col("PAYEMS").pct_change(1) * 100).alias("PAYEMS % chg 1M"),
        #(pl.col("PAYEMS").pct_change(3) * 100).alias("PAYEMS % chg 3M"),
        #(pl.col("PAYEMS").pct_change(6) * 100).alias("PAYEMS % chg 6M"),
        (pl.col("PAYEMS").pct_change(12) * 100).alias("PAYEMS % chg 12M"),
        #(pl.col("PAYEMS").pct_change(24) * 100).alias("PAYEMS % chg 24M"),
    ]).drop("PAYEMS")

    # Join all dataframes on 'observation_date' using "inner" joins to only keep dates where we have all 3 signals
    df = (
        FEDFUNDS
        .join(bbk, on="observation_date", how="inner")
        #.join(BBKMCOIX, on="observation_date", how="inner")
        .join(CORE, on="observation_date", how="inner")
        .join(m2real, on="observation_date", how="inner")
        .join(payems, on="observation_date", how="inner")
        .join(UNRATE, on="observation_date", how="inner")
    )
    # We want to predict FEDFUNDS at time T+1 given data at time T.
    # So we shift the FEDFUNDS column backward by 1 to create the label for the current row.
    df = df.with_columns(
        pl.col("FEDFUNDS").shift(-1).alias("target_next_month")
    )
    # Drop rows with nulls (caused by the shift at the end, or missing UNRATE data)
    df = df.drop_nulls()
    # Sort by date to ensure time-sequence is preserved
    df = df.sort("observation_date")
    return df


if __name__ == "__main__":
    set_seed()

    train_df = prepare_data()
    feature_cols = [c for c in train_df.columns if c not in ["observation_date", "target_next_month"]]
    target_col = "target_next_month"
    print(train_df)
    print(f"Feature count: {len(feature_cols)}")

    # Extract values
    X_all = train_df.select(feature_cols).to_numpy()
    y_all = train_df.select(target_col).to_numpy()

    # --- Configuration ---
    TRAIN_SPLIT_RATIO = 0.8
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 200

    # --- Train/Test split (Time-based, NOT random shuffle) ---
    split_idx = int(len(train_df) * TRAIN_SPLIT_RATIO)
    X_train_raw, X_test_raw = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    # --- Normalization ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)  # Fit scaler ONLY on training data to avoid leakage.
    X_test = scaler.transform(X_test_raw)

    # --- Data loader and model init ---
    train_loader = DataLoader(FedFundsDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(FedFundsDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = FedRatePredictor(input_dim=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training loop ---
    train_losses = []
    test_losses = []
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(test_loader)

        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    print("Training complete.")

    model.eval()
    with torch.no_grad():
        # Predict on the full test set
        test_inputs = torch.tensor(X_test, dtype=torch.float32)
        predictions = model(test_inputs).numpy()

    # Get dates for the test set for plotting
    # We slice the original data to match the test split index
    test_dates = train_df.select("observation_date")[split_idx:].to_series().to_list()

    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, label='Actual Fed Funds Rate', color='black', alpha=0.7)
    plt.plot(test_dates, predictions, label='Predicted (Next Month)', color='blue', linestyle='--')
    plt.title("Fed Funds Rate Prediction (OOS Test Data)")
    plt.xlabel("Date")
    plt.ylabel("Rate (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
