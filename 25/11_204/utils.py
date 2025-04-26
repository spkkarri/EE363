import os
import uuid
import scipy.io
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use('Agg')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_soh_per_battery(filenames):
    soh_dict = {}
    for file in filenames:
        data = scipy.io.loadmat(file)
        battery_name = file.split("/")[-1].split(".")[0]
        battery = data[list(data.keys())[-1]][0][0]
        cycles = battery['cycle'][0]
        battery_soh = [c['data'][0, 0]['Capacity'][0][0] for c in cycles if c['type'][0] == 'discharge']
        soh_dict[battery_name] = np.array(battery_soh, dtype=np.float32)
    return soh_dict

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

def preprocess_data(soh_array, seq_len=20):
    soh_min, soh_max = soh_array.min(), soh_array.max()
    soh_norm = (soh_array - soh_min) / (soh_max - soh_min)
    X, y = create_sequences(soh_norm, seq_len)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return X_tensor, y_tensor, dataloader, soh_min, soh_max

def train_and_evaluate(model, dataloader, X_tensor, y_tensor, soh_min, soh_max, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy()
    true = y_tensor.cpu().numpy()
    pred_soh = preds * (soh_max - soh_min) + soh_min
    true_soh = true * (soh_max - soh_min) + soh_min
    metrics = {
        "MAE": float(mean_absolute_error(true_soh, pred_soh)),
        "RMSE": float(np.sqrt(mean_squared_error(true_soh, pred_soh))),
        "R2": float(r2_score(true_soh, pred_soh))
    }
    return pred_soh.tolist(), true_soh.tolist(), metrics

def generate_plots(true_soh, pred_soh, output_dir="static/plots"):
    os.makedirs(output_dir, exist_ok=True)
    uid = str(uuid.uuid4())

    plt.figure(figsize=(8, 4))
    plt.plot(true_soh, label="True SOH", marker='o')
    plt.plot(pred_soh, label="Predicted SOH", marker='x')
    plt.xlabel("Cycle")
    plt.ylabel("SOH")
    plt.title("SOH Prediction")
    plt.legend()
    soh_path = f"{output_dir}/soh_{uid}.png"
    plt.savefig(soh_path)
    plt.close()

    rul_true = [max(true_soh) - soh for soh in true_soh]
    rul_pred = [max(true_soh) - soh for soh in pred_soh]
    plt.figure(figsize=(8, 4))
    plt.plot(rul_true, label="True RUL", marker='o')
    plt.plot(rul_pred, label="Predicted RUL", marker='x')
    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.title("Remaining Useful Life (RUL) Prediction")
    plt.legend()
    rul_path = f"{output_dir}/rul_{uid}.png"
    plt.savefig(rul_path)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(true_soh, bins=20, alpha=0.5, label='True SOH')
    plt.hist(pred_soh, bins=20, alpha=0.5, label='Predicted SOH')
    plt.xlabel("SOH")
    plt.ylabel("Frequency")
    plt.title("SOH Distribution (True vs Predicted)")
    plt.legend()
    hist_path = f"{output_dir}/histogram_{uid}.png"
    plt.savefig(hist_path)
    plt.close()

    diff = np.abs(np.array(true_soh) - np.array(pred_soh))
    plt.figure(figsize=(8, 4))
    sns.heatmap([diff], cmap='Blues', annot=False)
    plt.xlabel("Cycle Index")
    plt.ylabel("SOH Difference")
    plt.title("Difference Heatmap (True vs Predicted)")
    heatmap_path = f"{output_dir}/heatmap_{uid}.png"
    plt.savefig(heatmap_path)
    plt.close()

    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Value": [round(mean_absolute_error(true_soh, pred_soh), 4),
                  round(np.sqrt(mean_squared_error(true_soh, pred_soh)), 4),
                  round(r2_score(true_soh, pred_soh), 4)]
    })
    table_path = f"{output_dir}/metrics_table_{uid}.png"
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, cellLoc='center', loc='center')
    plt.savefig(table_path)
    plt.close()

    return soh_path, rul_path, hist_path, heatmap_path, table_path

def save_plots(true_t, pred_t, pred_l, residuals, conf_matrix, soh_data):
    plt.figure(figsize=(10, 5))
    plt.plot(true_t, label='Actual SOH', color='black')
    plt.plot(pred_t, label='Transformer', alpha=0.7)
    plt.plot(pred_l, label='LSTM+CNN', alpha=0.7)
    plt.title("📊 Model Comparison: SOH Prediction")
    plt.xlabel("Cycle Index")
    plt.ylabel("SOH")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/plots/soh_comparison.png')
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(residuals, label="Residuals", color="red")
    plt.axhline(0, color='black', linestyle='--')
    plt.title("🧼 Transformer Residual Plot")
    plt.xlabel("Cycle Index")
    plt.ylabel("Error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/plots/residuals_plot.png')
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(true_t, pred_t, alpha=0.6, color='green')
    plt.plot([min(true_t), max(true_t)], [min(true_t), max(true_t)], 'k--')
    plt.xlabel("Actual SOH")
    plt.ylabel("Predicted SOH")
    plt.title("📌 Actual vs Predicted SOH (Transformer)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/plots/scatter_plot.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, color="orange", edgecolor="black")
    plt.title("📊 Error Distribution")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig('static/plots/error_histogram.png')
    plt.close()

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("🧼 Confusion Matrix (Binned SOH)")
    plt.xlabel("Predicted Bin")
    plt.ylabel("Actual Bin")
    plt.tight_layout()
    plt.savefig('static/plots/confusion_matrix.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    for battery, soh in soh_data.items():
        plt.plot(soh, label=battery)
    plt.title("🔋 Capacity Curves for Each Battery")
    plt.xlabel("Cycle Index")
    plt.ylabel("Capacity (SOH)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('static/plots/capacity_curves.png')
    plt.close()

def extract_capacity_and_predictions(model, inputs, true_soh, output_dir='static/plots'):
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).squeeze().cpu().numpy()
    true_soh = true_soh.cpu().numpy()

    capacity = [s * 100 for s in true_soh]
    plt.figure()
    plt.plot(capacity, label='Battery Capacity (%)', color='blue')
    plt.xlabel('Cycle')
    plt.ylabel('Capacity (%)')
    plt.title('Battery Capacity Curve')
    plt.legend()
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    capacity_plot_path = os.path.join(output_dir, 'capacity_curve.png')
    plt.savefig(capacity_plot_path)
    plt.close()

    prediction_table = [
        {
            "Cycle": i + 1,
            "Actual SOH": round(true_val * 100, 2),
            "Predicted SOH": round(pred_val * 100, 2),
            "Error (%)": round(abs(pred_val - true_val) * 100, 2)
        }
        for i, (true_val, pred_val) in enumerate(zip(true_soh, predictions))
    ]
    return 'capacity_curve.png', prediction_table
