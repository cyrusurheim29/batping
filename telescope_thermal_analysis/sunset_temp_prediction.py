#!/usr/bin/env python3
"""
Sunset Temperature Prediction using Machine Learning

Predicts the temperature at sunset using time series history from the start
of the day up to 3 hours before sunset.

Train/Test split: Even days (training), Odd days (testing)
Metric: Mean Absolute Error (MAE)
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Check for optional dependencies
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: xgboost not installed. Using sklearn GradientBoosting instead.")
    from sklearn.ensemble import GradientBoostingRegressor

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. LSTM method will be skipped.")

# =============================================================================
# CONFIGURATION
# =============================================================================
LAT, LON, ELEVATION = -30.24, -70.74, 2663
HOURS_BEFORE_SUNSET = 3
SAMPLES_PER_HOUR = 4  # 15-minute intervals

site = EarthLocation(lat=LAT*u.deg, lon=LON*u.deg, height=ELEVATION*u.m)

# =============================================================================
# DATA PREPARATION
# =============================================================================

def get_sunset_time(date):
    """Calculate sunset time for a given date at Cerro Pachón."""
    start = Time(f"{date} 20:00:00")
    times = start + np.arange(0, 360, 1) * u.min
    altaz_frame = AltAz(obstime=times, location=site)
    sun_alt = get_sun(times).transform_to(altaz_frame).alt.deg

    for i in range(1, len(sun_alt)):
        if sun_alt[i-1] > -0.833 and sun_alt[i] <= -0.833:
            frac = (-0.833 - sun_alt[i-1]) / (sun_alt[i] - sun_alt[i-1])
            return (times[i-1] + frac * u.min).datetime
    return None


def prepare_dataset(df):
    """
    Prepare dataset for sunset temperature prediction.

    Returns list of (date, input_temps, sunset_temp, sunset_time) tuples.
    """
    df = df.copy()
    df['date'] = df['timestamp'].dt.date

    unique_dates = sorted(df['date'].unique())
    dataset = []

    print("Preparing dataset (calculating sunset times)...")
    for date in unique_dates:
        sunset = get_sunset_time(date)
        if sunset is None:
            continue

        sunset_ts = pd.Timestamp(sunset)
        cutoff_ts = sunset_ts - timedelta(hours=HOURS_BEFORE_SUNSET)
        day_start = pd.Timestamp(f"{date} 00:00:00")

        # Get data from start of day to cutoff (3h before sunset)
        day_data = df[(df['timestamp'] >= day_start) & (df['timestamp'] <= cutoff_ts)]

        # Get temperature at sunset (nearest measurement)
        sunset_data = df[(df['timestamp'] - sunset_ts).abs() < timedelta(minutes=10)]

        if len(day_data) < 20 or len(sunset_data) == 0:
            continue

        input_temps = day_data['temperature'].values
        sunset_temp = sunset_data.iloc[(sunset_data['timestamp'] - sunset_ts).abs().argmin()]['temperature']

        dataset.append({
            'date': date,
            'input_temps': input_temps,
            'sunset_temp': sunset_temp,
            'sunset_time': sunset_ts,
            'cutoff_time': cutoff_ts,
            'n_samples': len(input_temps)
        })

    return dataset


def split_train_test(dataset):
    """Split by even/odd days of month."""
    train = [d for d in dataset if d['date'].day % 2 == 0]
    test = [d for d in dataset if d['date'].day % 2 == 1]
    return train, test


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def extract_features(temps):
    """
    Extract features from temperature time series for ML models.
    """
    if len(temps) == 0:
        return np.zeros(20)

    features = []

    # Basic statistics
    features.append(temps[-1])  # Most recent temp (3h before sunset)
    features.append(np.mean(temps))
    features.append(np.std(temps))
    features.append(np.min(temps))
    features.append(np.max(temps))
    features.append(temps[-1] - temps[0])  # Total change during day

    # Recent values (last 3 hours in 1-hour chunks if available)
    n = len(temps)
    for hours_ago in [0, 1, 2]:
        idx = n - 1 - hours_ago * SAMPLES_PER_HOUR
        if idx >= 0:
            features.append(temps[idx])
        else:
            features.append(temps[0])

    # Recent trend (last 2 hours)
    recent_n = min(2 * SAMPLES_PER_HOUR, n)
    recent = temps[-recent_n:]
    if len(recent) > 1:
        trend = (recent[-1] - recent[0]) / len(recent)
        features.append(trend)
    else:
        features.append(0)

    # Morning vs afternoon mean
    mid = n // 2
    if mid > 0:
        features.append(np.mean(temps[:mid]))  # Earlier part of day
        features.append(np.mean(temps[mid:]))  # Later part of day
    else:
        features.append(temps[0])
        features.append(temps[-1])

    # Temperature velocity (recent derivative)
    if n >= 2:
        velocity = np.diff(temps[-min(8, n):]).mean()
        features.append(velocity)
    else:
        features.append(0)

    # Acceleration (change in velocity)
    if n >= 4:
        recent_diff = np.diff(temps[-min(8, n):])
        if len(recent_diff) >= 2:
            accel = recent_diff[-1] - recent_diff[0]
            features.append(accel)
        else:
            features.append(0)
    else:
        features.append(0)

    # Percentiles
    features.append(np.percentile(temps, 25))
    features.append(np.percentile(temps, 75))

    # Number of samples (proxy for time of year / day length)
    features.append(n)

    # Range in last 3 hours
    last_3h = temps[-min(3*SAMPLES_PER_HOUR, n):]
    features.append(np.max(last_3h) - np.min(last_3h))

    return np.array(features)


def prepare_features(dataset):
    """Prepare feature matrix and targets."""
    X = np.array([extract_features(d['input_temps']) for d in dataset])
    y = np.array([d['sunset_temp'] for d in dataset])
    return X, y


# =============================================================================
# METHOD 1: BASELINE (3h before sunset temperature)
# =============================================================================

def baseline_predict(dataset):
    """Simply use the temperature at 3 hours before sunset."""
    predictions = [d['input_temps'][-1] for d in dataset]
    actuals = [d['sunset_temp'] for d in dataset]
    return np.array(predictions), np.array(actuals)


# =============================================================================
# METHOD 2: LINEAR REGRESSION WITH FEATURES
# =============================================================================

def train_linear_model(X_train, y_train):
    """Train Ridge regression model."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y_train)
    return model, scaler


def predict_linear(model, scaler, X):
    """Predict with linear model."""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


# =============================================================================
# METHOD 3: GRADIENT BOOSTING
# =============================================================================

def train_gbm_model(X_train, y_train):
    """Train gradient boosting model."""
    if HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    model.fit(X_train, y_train)
    return model


# =============================================================================
# METHOD 4: LSTM NEURAL NETWORK
# =============================================================================

class LSTMModel(nn.Module):
    def __init__(self, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


def pad_sequences(sequences, max_len):
    """Pad sequences to same length."""
    padded = np.zeros((len(sequences), max_len))
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, -length:] = seq[-length:]  # Right-align (most recent at end)
    return padded


def train_lstm_model(train_data, max_len=80, epochs=50, lr=0.001):
    """Train LSTM model."""
    if not HAS_TORCH:
        return None, None, None

    # Prepare sequences
    X_seqs = [d['input_temps'] for d in train_data]
    y = np.array([d['sunset_temp'] for d in train_data])

    # Pad and normalize
    X_padded = pad_sequences(X_seqs, max_len)

    # Normalize
    X_mean, X_std = X_padded.mean(), X_padded.std()
    y_mean, y_std = y.mean(), y.std()

    X_norm = (X_padded - X_mean) / (X_std + 1e-8)
    y_norm = (y - y_mean) / (y_std + 1e-8)

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_norm).unsqueeze(-1)  # (N, seq_len, 1)
    y_tensor = torch.FloatTensor(y_norm)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Train
    model = LSTMModel(hidden_size=32, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    norm_params = {'X_mean': X_mean, 'X_std': X_std, 'y_mean': y_mean, 'y_std': y_std}
    return model, norm_params, max_len


def predict_lstm(model, norm_params, test_data, max_len):
    """Predict with LSTM model."""
    if model is None:
        return None

    X_seqs = [d['input_temps'] for d in test_data]
    X_padded = pad_sequences(X_seqs, max_len)
    X_norm = (X_padded - norm_params['X_mean']) / (norm_params['X_std'] + 1e-8)
    X_tensor = torch.FloatTensor(X_norm).unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        pred_norm = model(X_tensor).numpy()

    # Denormalize
    pred = pred_norm * norm_params['y_std'] + norm_params['y_mean']
    return pred


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    print("="*70)
    print("SUNSET TEMPERATURE PREDICTION - ML MODEL COMPARISON")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('rubin_mirror_temps.csv', parse_dates=['timestamp'])
    print(f"Loaded {len(df)} temperature measurements")

    # Prepare dataset
    dataset = prepare_dataset(df)
    print(f"Prepared {len(dataset)} days with valid sunset data")

    # Split train/test
    train_data, test_data = split_train_test(dataset)
    print(f"Training set: {len(train_data)} days (even days of month)")
    print(f"Test set: {len(test_data)} days (odd days of month)")

    # Prepare feature matrices
    print("\nExtracting features...")
    X_train, y_train = prepare_features(train_data)
    X_test, y_test = prepare_features(test_data)
    print(f"Feature dimension: {X_train.shape[1]}")

    results = {}

    # -------------------------------------------------------------------------
    # METHOD 1: BASELINE
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("METHOD 1: BASELINE (Temperature 3h before sunset)")
    print("-"*70)

    pred_baseline, actual = baseline_predict(test_data)
    mae_baseline = mean_absolute_error(actual, pred_baseline)
    results['Baseline'] = mae_baseline
    print(f"MAE: {mae_baseline:.4f} °C")

    # -------------------------------------------------------------------------
    # METHOD 2: LINEAR REGRESSION
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("METHOD 2: RIDGE REGRESSION WITH ENGINEERED FEATURES")
    print("-"*70)

    linear_model, scaler = train_linear_model(X_train, y_train)
    pred_linear = predict_linear(linear_model, scaler, X_test)
    mae_linear = mean_absolute_error(y_test, pred_linear)
    results['Ridge Regression'] = mae_linear
    print(f"MAE: {mae_linear:.4f} °C")

    # Feature importance (absolute coefficients)
    feature_names = ['temp_3h_before', 'mean', 'std', 'min', 'max', 'total_change',
                     'temp_0h', 'temp_1h', 'temp_2h', 'trend_2h', 'early_mean',
                     'late_mean', 'velocity', 'acceleration', 'p25', 'p75',
                     'n_samples', 'range_3h']
    coefs = np.abs(linear_model.coef_)
    top_features = np.argsort(coefs)[-5:][::-1]
    print("Top 5 features by importance:")
    for idx in top_features:
        if idx < len(feature_names):
            print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")

    # -------------------------------------------------------------------------
    # METHOD 3: GRADIENT BOOSTING
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("METHOD 3: GRADIENT BOOSTING (XGBoost)" if HAS_XGB else "METHOD 3: GRADIENT BOOSTING (sklearn)")
    print("-"*70)

    gbm_model = train_gbm_model(X_train, y_train)
    pred_gbm = gbm_model.predict(X_test)
    mae_gbm = mean_absolute_error(y_test, pred_gbm)
    results['Gradient Boosting'] = mae_gbm
    print(f"MAE: {mae_gbm:.4f} °C")

    # -------------------------------------------------------------------------
    # METHOD 4: LSTM
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("METHOD 4: LSTM NEURAL NETWORK")
    print("-"*70)

    if HAS_TORCH:
        print("Training LSTM (this may take a moment)...")
        lstm_model, norm_params, max_len = train_lstm_model(train_data, epochs=50)
        pred_lstm = predict_lstm(lstm_model, norm_params, test_data, max_len)
        mae_lstm = mean_absolute_error(y_test, pred_lstm)
        results['LSTM'] = mae_lstm
        print(f"MAE: {mae_lstm:.4f} °C")
    else:
        print("Skipped (PyTorch not installed)")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON (Test Set MAE)")
    print("="*70)
    print(f"{'Method':<25} {'MAE (°C)':<12} {'vs Baseline':<15}")
    print("-"*70)

    baseline_mae = results['Baseline']
    for method, mae in sorted(results.items(), key=lambda x: x[1]):
        improvement = ((baseline_mae - mae) / baseline_mae) * 100
        sign = "+" if improvement > 0 else ""
        print(f"{method:<25} {mae:<12.4f} {sign}{improvement:.1f}%")

    # Best method
    best_method = min(results, key=results.get)
    best_mae = results[best_method]

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"\nBest performing method: {best_method}")
    print(f"Test MAE: {best_mae:.4f} °C")
    print(f"Improvement over baseline: {((baseline_mae - best_mae) / baseline_mae) * 100:.1f}%")

    # Save predictions for analysis
    pred_df = pd.DataFrame({
        'date': [d['date'] for d in test_data],
        'actual': y_test,
        'baseline': pred_baseline,
        'ridge': pred_linear,
        'gbm': pred_gbm,
    })
    if HAS_TORCH:
        pred_df['lstm'] = pred_lstm
    pred_df.to_csv('sunset_predictions.csv', index=False)
    print("\nPredictions saved to: sunset_predictions.csv")

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Actual vs Predicted (best model)
    ax1 = axes[0, 0]
    best_pred = pred_gbm if best_method == 'Gradient Boosting' else (pred_lstm if best_method == 'LSTM' else pred_linear)
    ax1.scatter(y_test, best_pred, alpha=0.5, s=30)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect prediction')
    ax1.set_xlabel('Actual Sunset Temperature (°C)')
    ax1.set_ylabel('Predicted Sunset Temperature (°C)')
    ax1.set_title(f'Best Model ({best_method}): Actual vs Predicted')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals over time
    ax2 = axes[0, 1]
    dates = [d['date'] for d in test_data]
    residuals = y_test - best_pred
    ax2.scatter(dates, residuals, alpha=0.5, s=30)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Residual (Actual - Predicted) °C')
    ax2.set_title('Prediction Residuals Over Time')
    ax2.grid(True, alpha=0.3)

    # Plot 3: MAE comparison bar chart
    ax3 = axes[1, 0]
    methods = list(results.keys())
    maes = [results[m] for m in methods]
    colors = ['green' if m == best_method else 'steelblue' for m in methods]
    bars = ax3.bar(methods, maes, color=colors)
    ax3.set_ylabel('Mean Absolute Error (°C)')
    ax3.set_title('Model Comparison: MAE on Test Set')
    ax3.tick_params(axis='x', rotation=15)
    for bar, mae in zip(bars, maes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mae:.3f}', ha='center', va='bottom', fontsize=10)

    # Plot 4: Residual histogram
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=25, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', label='Zero error')
    ax4.axvline(x=residuals.mean(), color='orange', linestyle='-', label=f'Mean: {residuals.mean():.2f}°C')
    ax4.set_xlabel('Prediction Error (°C)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Distribution of Prediction Errors ({best_method})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ml_model_comparison.png', dpi=150)
    print("Visualization saved to: ml_model_comparison.png")


if __name__ == "__main__":
    main()
