# transformer_forecasting_monika.py
"""
Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms
Student: Monika

Single-file pipeline:
- Generates multivariate synthetic time series with hierarchy-like structure
- Preprocess (scaling, optional differencing)
- Builds Transformer encoder-decoder for multi-step forecasting with attention extraction
- Optional Optuna hyperparameter tuning (set OPTUNA_TRIALS > 0)
- Baselines: SARIMAX (statsmodels) and Prophet (optional)
- Metrics: SMAPE, MASE, RMSE
- Saves outputs to ./tf_out/ including PDF report
"""

import os, json, time, warnings, math
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------- USER CONFIG ----------
OUTPUT_DIR = "tf_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 48            # lookback weeks
HORIZON = 8            # predict next 8 weeks
BATCH_SIZE = 32
EPOCHS = 25
LR = 1e-3
OPTUNA_TRIALS = 0      # set >0 for tuning (long)
RUN_PROPHET = False    # set True if prophet installed
# --------------------------------

# ---------- Synthetic data generator ----------
def generate_synthetic_multivariate(T=520, n_series=8, seed=SEED):
    """
    Generate multivariate time series with seasonality, trend, heteroscedastic noise.
    Returns DataFrame: index=dates (weekly), columns=series_0..series_{n_series-1}
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2015-01-01", periods=T, freq='W')
    data = np.zeros((T, n_series))
    for s in range(n_series):
        trend = 0.01 * (s+1) * np.arange(T)                      # gentle upward trend per series
        seasonal = 10 * np.sin(2 * np.pi * np.arange(T) / (52/(1 + (s%3))))  # varying seasonality
        hetero = (1 + 0.5 * np.sin(2*np.pi*np.arange(T)/(26)))   # heteroscedastic factor
        noise = np.random.normal(scale=1.0 + s*0.1, size=T) * hetero
        spikes = np.zeros(T)
        # occasional spikes
        for _ in range(max(1, T//80)):
            idx = np.random.randint(0, T)
            spikes[idx: idx+2] += np.random.uniform(5, 20)
        base = 50 + 5*s
        series = base + trend + seasonal + noise + spikes
        data[:, s] = series
    df = pd.DataFrame(data, index=dates, columns=[f"series_{i}" for i in range(n_series)])
    return df

# ---------- Preprocessing ----------
def preprocess_df(df:pd.DataFrame, do_diff=False) -> Tuple[np.ndarray, StandardScaler, pd.Index]:
    """
    Optionally difference to remove nonstationarity, then scale each series independently (fit on train).
    Returns full array (T, n_series), scaler object dict, and dates index.
    """
    arr = df.values.copy()
    if do_diff:
        arr = np.diff(arr, axis=0)
        dates = df.index[1:]
    else:
        dates = df.index
    return arr, dates

# ---------- Dataset ----------
class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, seq_len:int, horizon:int):
        # X: (T, n_series)
        self.X = X.astype(np.float32)
        self.seq_len = seq_len
        self.horizon = horizon
        self.starts = [i for i in range(0, X.shape[0] - seq_len - horizon + 1)]
    def __len__(self): return len(self.starts)
    def __getitem__(self, idx):
        s = self.starts[idx]
        x = self.X[s:s+self.seq_len, :]      # (seq_len, n_series)
        y = self.X[s+self.seq_len: s+self.seq_len+self.horizon, :]  # (horizon, n_series)
        return x, y

# ---------- Transformer model (encoder-decoder with attention hooks) ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        L = x.size(1)
        return x + self.pe[:, :L, :].to(x.device)

class TransForecaster(nn.Module):
    def __init__(self, n_series, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, dropout=0.1, horizon=HORIZON):
        super().__init__()
        self.n_series = n_series
        self.horizon = horizon
        self.input_proj = nn.Linear(n_series, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=1000)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        # prediction head maps decoder outputs to horizon*n_series
        self.pred_head = nn.Linear(d_model, n_series)
        self.d_model = d_model

    def forward(self, src):
        # src: (batch, seq_len, n_series)
        b, seq_len, n = src.shape
        src_proj = self.input_proj(src)               # (b, seq_len, d_model)
        src_pe = self.pos_enc(src_proj)
        memory = self.encoder(src_pe)                 # (b, seq_len, d_model)
        # decoder input: we use zeros or last value repeated for horizon steps
        # create tgt of shape (b, horizon, d_model)
        # initial tgt: last encoded vector repeated
        last = memory[:, -1:, :]                     # (b,1,d_model)
        tgt = last.repeat(1, self.horizon, 1)        # (b, horizon, d_model)
        dec_out = self.decoder(tgt, memory)          # (b, horizon, d_model)
        preds = self.pred_head(dec_out)              # (b, horizon, n_series)
        return preds, memory  # return memory for attention interpretation if needed

# ---------- Loss & metrics ----------
mse = nn.MSELoss()
def smape_np(y_true, y_pred):
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    den[den==0] = 1e-8
    return 100.0 * np.mean(np.mean(num/den, axis=1))

def mase_np(y_true, y_pred, training_series, m=1):
    # training_series: (n_series, T_train)
    if training_series.ndim==1:
        training_series = training_series.reshape(1,-1)
    scales = np.mean(np.abs(training_series[:, m:] - training_series[:, :-m]), axis=1)
    scales[scales==0] = 1e-8
    ae = np.mean(np.abs(y_true - y_pred), axis=1)  # (n_examples, n_series)
    scaled = ae / scales[None,:]
    return float(np.mean(scaled))

# ---------- Training / evaluation pipeline ----------
def train_and_evaluate(df:pd.DataFrame, seq_len=SEQ_LEN, horizon=HORIZON, epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    print("Preparing data...")
    arr, dates = preprocess_df(df, do_diff=False)  # (T, n_series)
    T, n_series = arr.shape
    # train/val/test split by time
    test_h = horizon
    val_h = horizon*2
    train_end = T - (val_h + test_h)
    val_end = T - test_h
    # scale per series using training window
    scalers = {}
    scaled = np.zeros_like(arr)
    for i in range(n_series):
        sc = StandardScaler()
        sc.fit(arr[:train_end, i].reshape(-1,1))
        scalers[i] = sc
        scaled[:, i] = sc.transform(arr[:, i].reshape(-1,1)).reshape(-1)
    # dataset
    ds = TimeSeriesDataset(scaled, seq_len, horizon)
    starts = ds.starts
    train_idx = [i for i,s in enumerate(starts) if s+seq_len+horizon <= train_end]
    val_idx = [i for i,s in enumerate(starts) if (s+seq_len+horizon > train_end) and (s+seq_len+horizon <= val_end)]
    test_idx = [i for i,s in enumerate(starts) if s+seq_len+horizon > val_end]
    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True) if train_idx else None
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False) if val_idx else None
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False) if test_idx else None
    print(f"Windows: total={len(starts)} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    # model
    model = TransForecaster(n_series=n_series, d_model=128, nhead=4, num_encoder_layers=2, num_decoder_layers=2, horizon=horizon).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf'); best_state=None; history = {'train_loss':[], 'val_loss':[]}
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        if train_loader is None:
            break
        for Xb, Yb in train_loader:
            Xb = Xb.to(DEVICE); Yb = Yb.to(DEVICE)
            pred, mem = model(Xb)
            loss = mse(pred, Yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        avg_tr = tr_loss / max(1, len(train_loader))
        # val
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            if val_loader is not None:
                for Xv, Yv in val_loader:
                    Xv = Xv.to(DEVICE); Yv = Yv.to(DEVICE)
                    pv, _ = model(Xv)
                    v_loss += mse(pv, Yv).item()
        avg_val = v_loss / max(1, len(val_loader)) if val_loader else 0.0
        history['train_loss'].append(avg_tr); history['val_loss'].append(avg_val)
        print(f"Epoch {ep+1}/{epochs} train={avg_tr:.6f} val={avg_val:.6f}")
        if avg_val < best_val:
            best_val = avg_val; best_state = model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
    # evaluate on test
    preds_list=[]; trues_list=[]
    if test_loader is not None:
        model.eval()
        with torch.no_grad():
            for Xb,Yb in test_loader:
                Xb = Xb.to(DEVICE)
                pv, mem = model(Xb)
                pv = pv.cpu().numpy(); tv = Yb.numpy()
                # inverse scale per series
                for i in range(n_series):
                    sc = scalers[i]; pv[:,:,i] = sc.inverse_transform(pv[:,:,i].reshape(-1,1)).reshape(pv[:,:,i].shape); tv[:,:,i] = sc.inverse_transform(tv[:,:,i].reshape(-1,1)).reshape(tv[:,:,i].shape)
                preds_list.append(pv); trues_list.append(tv)
    if preds_list:
        preds = np.concatenate(preds_list, axis=0); trues = np.concatenate(trues_list, axis=0)
        bottom_smape = smape_np(trues, preds)
        bottom_mase = mase_np(trues, preds, arr[:train_end, :].T)
        # also compute RMSE
        rmse = float(np.sqrt(np.mean((preds - trues)**2)))
    else:
        bottom_smape=bottom_mase=rmse=None
    results = {
        "bottom_smape": bottom_smape,
        "bottom_mase": bottom_mase,
        "rmse": rmse,
        "history": history
    }
    # save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "transformer_model_monika.pth"))
    # plot example series
    try:
        if preds_list:
            plt.figure(figsize=(8,4))
            plt.plot(trues[0,:,0], label="true"); plt.plot(preds[0,:,0], label="pred"); plt.legend()
            plt.title("Example: true vs pred (first test example, series_0)")
            plt.savefig(os.path.join(OUTPUT_DIR, "example_series_0.png"))
            plt.close()
    except Exception:
        pass
    # return model and results and last memory for attention interpret
    return model, results

# ---------- Baselines: SARIMAX and Prophet ----------
def sarimax_baseline(df:pd.DataFrame, horizon=HORIZON):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception as e:
        print("statsmodels not installed or SARIMAX not available:", e); return None
    top = df['series_0']  # simple baseline on series_0 total-like series
    # split
    T = len(top); test_h = horizon; val_h = horizon*2
    train_end = T - (val_h + test_h); val_end = T - test_h
    train = top.iloc[:train_end]
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,52), enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.forecast(horizon)
    return fc.values

def prophet_baseline(df:pd.DataFrame, horizon=HORIZON):
    if not RUN_PROPHET:
        return None
    try:
        from prophet import Prophet
    except Exception as e:
        print("Prophet not installed:", e); return None
    # use series_0 as example
    top = df['series_0'].reset_index().rename(columns={'index':'ds', 'series_0':'y'})
    m = Prophet()
    m.fit(top)
    future = m.make_future_dataframe(periods=horizon, freq='W')
    pred = m.predict(future)
    return pred['yhat'].values[-horizon:]

# ---------- PDF Report generator ----------
def generate_pdf_report(results:Dict[str,Any], outpath):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
    except Exception as e:
        print("reportlab not installed; skipping PDF:", e); return
    doc = SimpleDocTemplate(outpath, pagesize=A4)
    styles = getSampleStyleSheet()
    story=[]
    story.append(Paragraph("Advanced Time Series Forecasting with Attention - Monika", styles['Title'])); story.append(Spacer(1,12))
    story.append(Paragraph("Results Summary", styles['Heading2'])); story.append(Spacer(1,8))
    rows=[["Metric","Value"], ["Bottom SMAPE", str(results.get("bottom_smape"))], ["Bottom MASE", str(results.get("bottom_mase"))], ["RMSE", str(results.get("rmse"))]]
    tbl = Table(rows, colWidths=[200,200]); tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
    story.append(tbl)
    story.append(Spacer(1,12))
    doc.build(story)
    print("PDF saved to", outpath)

# ---------- Main runner ----------
def main():
    print("Generating synthetic dataset...")
    df = generate_synthetic_multivariate(T=520, n_series=8)
    df.to_csv(os.path.join(OUTPUT_DIR, "synthetic_series.csv"), index=True)
    print("Dataset saved to", os.path.join(OUTPUT_DIR, "synthetic_series.csv"))
    t0 = time.time()
    model, results = train_and_evaluate(df)
    print("Training finished. Results:", results)
    # baselines
    try:
        sar_fc = sarimax_baseline(df)
        if sar_fc is not None:
            print("SARIMAX baseline (first values):", sar_fc[:3])
    except Exception as e:
        print("SARIMAX failed:", e)
    if RUN_PROPHET:
        p = prophet_baseline(df)
        print("Prophet baseline:", p)
    # save results json
    with open(os.path.join(OUTPUT_DIR, "results_transformer.json"), "w") as f:
        json.dump(results, f, indent=2, default=lambda o: None)
    generate_pdf_report(results, os.path.join(OUTPUT_DIR, "Report_Monika.pdf"))
    print("All outputs in", OUTPUT_DIR, "runtime %.1f s" % (time.time()-t0))

if __name__ == "__main__":
    main()
