import torch
import torch.nn as nn

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transformer Model for SOH Prediction ===
class TransformerSOHPredictor(nn.Module):
    def __init__(self, seq_len, model_dim=64, num_layers=2, nhead=4, pred_len=1):
        super(TransformerSOHPredictor, self).__init__()
        self.input_projection = nn.Linear(seq_len, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=nhead, dim_feedforward=128
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(model_dim, pred_len)

    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(1) + self.positional_encoding[:, :1]
        x = x.permute(1, 0, 2)  # For transformer input: (seq_len, batch, feature)
        x = self.transformer_encoder(x)
        out = self.decoder(x[-1])  # Last token output
        return out.squeeze()


# === LSTM + CNN Hybrid Model for SOH Prediction ===
class LSTM_CNN_SOHPredictor(nn.Module):
    def __init__(self, seq_len, hidden_dim=64, cnn_out=32):
        super(LSTM_CNN_SOHPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_dim, batch_first=True
        )

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, cnn_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(hidden_dim + cnn_out, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # Add feature dim
        lstm_out, _ = self.lstm(x)
        lstm_feat = lstm_out[:, -1, :]

        cnn_feat = self.conv(x.permute(0, 2, 1)).squeeze(-1)
        combined = torch.cat((lstm_feat, cnn_feat), dim=1)

        out = self.fc(combined)
        return out.squeeze()


# === Model Selector ===
def get_model(model_name, seq_len):
    if model_name.lower() == 'transformer':
        return TransformerSOHPredictor(seq_len).to(device)
    elif model_name.lower() == 'lstmcnn':
        return LSTM_CNN_SOHPredictor(seq_len).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
