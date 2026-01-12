import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# Konstanta proyek
# =========================
SEED = 42
TARGET = "flow"

FEATURES_CAT = ["detid", "hour", "dow", "month", "is_weekend"]
FEATURES_NUM = ["occ", "dayofyear", "error_flag"]

DEFAULT_VAL_DAYS = 30
DEFAULT_TEST_DAYS = 30
DEFAULT_MAX_TRAIN_ROWS = 900_000


# =========================
# Util: metrik regresi
# =========================
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)  # kompatibel lintas versi sklearn
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-6))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}


# =========================
# Util: baseline median
# =========================
def median_baseline(train_df: pd.DataFrame, df_to_pred: pd.DataFrame) -> np.ndarray:
    key1 = ["detid", "dow", "hour"]
    key2 = ["detid", "hour"]

    g1 = train_df.groupby(key1)[TARGET].median().reset_index().rename(columns={TARGET: "pred"})
    g2 = train_df.groupby(key2)[TARGET].median().reset_index().rename(columns={TARGET: "pred2"})
    global_med = float(train_df[TARGET].median())

    tmp = df_to_pred.merge(g1, how="left", on=key1)
    tmp = tmp.merge(g2, how="left", on=key2)
    pred = tmp["pred"].fillna(tmp["pred2"]).fillna(global_med).to_numpy(dtype="float64")
    return pred


# =========================
# Load data + feature engineering
# =========================
@st.cache_data(show_spinner=False)
def read_raw_csv(uploaded_file):
    """
    Prioritas:
    1) Jika ada file di uploader -> baca dari situ
    2) Jika tidak, coba baca paris.csv yang ada di folder repo (kalau memang kamu commit)
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "upload"
        return df, source

    if os.path.exists("paris.csv"):
        df = pd.read_csv("paris.csv")
        source = "repo"
        return df, source

    return None, None


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop kolom yang tidak dipakai/bermasalah (speed = semuanya NaN di file ini)
    if "speed" in df.columns:
        df = df.drop(columns=["speed"])

    # Konversi tanggal
    df["day"] = pd.to_datetime(df["day"], format="%Y-%m-%d", errors="coerce")

    # interval: detik sejak tengah malam (0, 3600, ..., 82800)
    df["hour"] = (df["interval"] // 3600).astype("int16")

    # Timestamp lengkap
    df["timestamp"] = df["day"] + pd.to_timedelta(df["interval"], unit="s")

    # Fitur waktu
    df["dow"] = df["day"].dt.dayofweek.astype("int16")  # 0=Senin ... 6=Minggu
    df["month"] = df["day"].dt.month.astype("int16")
    df["dayofyear"] = df["day"].dt.dayofyear.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int16")

    # Flag error
    if "error" in df.columns:
        df["error_flag"] = df["error"].fillna(0).astype("int16")
        df = df.drop(columns=["error"])
    else:
        df["error_flag"] = 0

    # Pastikan kolom target ada dan tidak NaN
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan.")
    if df[TARGET].isna().any():
        df = df.dropna(subset=[TARGET])

    return df


def time_split(df: pd.DataFrame, val_days: int, test_days: int):
    df = df.sort_values("timestamp").reset_index(drop=True)

    max_day = df["day"].max()
    cut_test = max_day - pd.Timedelta(days=int(test_days))
    cut_val = cut_test - pd.Timedelta(days=int(val_days))

    train_df = df[df["day"] < cut_val].copy()
    val_df = df[(df["day"] >= cut_val) & (df["day"] < cut_test)].copy()
    test_df = df[df["day"] >= cut_test].copy()

    return train_df, val_df, test_df, cut_val, cut_test


# =========================
# Model: pipeline SGDRegressor + OneHot
# =========================
def build_pipeline(seed: int = SEED):
    # Robust terhadap versi sklearn berbeda
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", ohe, FEATURES_CAT),
            ("num", num_pipe, FEATURES_NUM),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # Robust loss: beberapa versi lama pakai 'squared_loss'
    try:
        reg = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-3,
            random_state=seed,
        )
    except Exception:
        reg = SGDRegressor(
            loss="squared_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-3,
            random_state=seed,
        )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", reg),
    ])
    return pipe


def ensure_trained_model(train_df, val_df, max_train_rows: int):
    rng = np.random.default_rng(SEED)

    if len(train_df) > max_train_rows:
        train_fit = train_df.sample(int(max_train_rows), random_state=SEED)
    else:
        train_fit = train_df

    X_train = train_fit[FEATURES_CAT + FEATURES_NUM]
    y_train = train_fit[TARGET].astype("float64")

    X_val = val_df[FEATURES_CAT + FEATURES_NUM]
    y_val = val_df[TARGET].astype("float64")

    pipe = build_pipeline(SEED)
    pipe.fit(X_train, y_train)

    val_pred = pipe.predict(X_val)
    metrics = regression_metrics(y_val, val_pred)

    return pipe, metrics


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Prediksi Flow Lalu Lintas (Paris)", layout="wide")

st.title("Prediksi Volume Lalu Lintas (flow) - Streamlit Demo")

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload paris.csv", type=["csv"])
    val_days = st.number_input("Val window (hari)", min_value=7, max_value=120, value=DEFAULT_VAL_DAYS, step=1)
    test_days = st.number_input("Test window (hari)", min_value=7, max_value=120, value=DEFAULT_TEST_DAYS, step=1)
    max_train_rows = st.number_input("Maks baris training", min_value=50_000, max_value=2_000_000, value=DEFAULT_MAX_TRAIN_ROWS, step=50_000)

    st.header("Aksi")
    do_train = st.button("Train model (SGDRegressor)")
    do_baseline = st.button("Hitung baseline median")

tab1, tab2, tab3 = st.tabs(["EDA & Split", "Evaluasi", "Prediksi Satu Titik"])

raw_df, source = read_raw_csv(uploaded)

if raw_df is None:
    st.error("Dataset belum tersedia. Upload file paris.csv atau commit paris.csv di repo (sejajar dengan app.py).")
    st.stop()

df = feature_engineering(raw_df)

train_df, val_df, test_df, cut_val, cut_test = time_split(df, val_days=int(val_days), test_days=int(test_days))

with tab1:
    st.subheader("Ringkasan dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sumber", "Upload" if source == "upload" else "Repo paris.csv")
    c2.metric("Total baris", f"{len(df):,}")
    c3.metric("Detektor unik (detid)", f"{df['detid'].nunique():,}")
    c4.metric("Rentang tanggal", f"{df['day'].min().date()} → {df['day'].max().date()}")

    st.write("Batas split berbasis waktu:")
    st.code(f"cut_val  = {cut_val.date()}\ncut_test = {cut_test.date()}")

    st.write("Ukuran split:")
    st.dataframe(pd.DataFrame({
        "split": ["train", "val", "test"],
        "rows": [len(train_df), len(val_df), len(test_df)]
    }))

    st.write("Contoh data (setelah feature engineering):")
    st.dataframe(df[["timestamp", "day", "interval", "detid", "hour", "dow", "month", "dayofyear", "is_weekend", "occ", "error_flag", TARGET]].head(20))

with tab2:
    st.subheader("Evaluasi baseline & model")

    # Baseline
    if do_baseline or ("baseline_test_pred" in st.session_state):
        with st.spinner("Menghitung baseline median..."):
            val_pred_base = median_baseline(train_df, val_df)
            test_pred_base = median_baseline(train_df, test_df)
            st.session_state["baseline_val_pred"] = val_pred_base
            st.session_state["baseline_test_pred"] = test_pred_base

        base_val_metrics = regression_metrics(val_df[TARGET], st.session_state["baseline_val_pred"])
        base_test_metrics = regression_metrics(test_df[TARGET], st.session_state["baseline_test_pred"])

        st.write("Baseline median per (detid, dow, hour) dengan fallback:")
        st.json({"Val": base_val_metrics, "Test": base_test_metrics})

    # Model
    if do_train or ("model" in st.session_state):
        if do_train:
            with st.spinner("Training model..."):
                model, val_metrics = ensure_trained_model(train_df, val_df, max_train_rows=int(max_train_rows))
                st.session_state["model"] = model
                st.session_state["model_val_metrics"] = val_metrics

        st.write("Metrik model di Validation:")
        st.json(st.session_state.get("model_val_metrics", {}))

        # Evaluasi test
        with st.spinner("Evaluasi di Test..."):
            X_test = test_df[FEATURES_CAT + FEATURES_NUM]
            y_test = test_df[TARGET].astype("float64")
            test_pred = st.session_state["model"].predict(X_test)
            st.session_state["model_test_pred"] = test_pred
            test_metrics = regression_metrics(y_test, test_pred)

        st.write("Metrik model di Test:")
        st.json(test_metrics)

        # Download prediksi test
        out = test_df[["timestamp", "detid", "hour", "dow", "occ", "error_flag", TARGET]].copy()
        out["pred_flow"] = st.session_state["model_test_pred"]
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download prediksi test (CSV)", data=csv_bytes, file_name="prediksi_flow_test.csv", mime="text/csv")

with tab3:
    st.subheader("Prediksi satu titik (input manual)")

    if "model" not in st.session_state:
        st.warning("Model belum ditrain. Buka tab 'Evaluasi' lalu klik 'Train model'.")
        st.stop()

    detid_list = sorted(df["detid"].unique().tolist())
    default_det = detid_list[0] if detid_list else 1

    c1, c2, c3 = st.columns(3)
    detid = c1.selectbox("detid", detid_list, index=0)
    d = c2.date_input("Tanggal", value=df["day"].max().date())
    hour = c3.number_input("Jam (0-23)", min_value=0, max_value=23, value=8, step=1)

    c4, c5 = st.columns(2)
    occ = c4.number_input("occ (occupancy)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    error_flag = c5.selectbox("error_flag", [0, 1], index=0)

    dts = pd.to_datetime(d)
    dow = int(dts.dayofweek)
    month = int(dts.month)
    dayofyear = int(dts.dayofyear)
    is_weekend = int(dow >= 5)

    X_one = pd.DataFrame([{
        "detid": detid,
        "hour": int(hour),
        "dow": dow,
        "month": month,
        "is_weekend": is_weekend,
        "occ": float(occ),
        "dayofyear": dayofyear,
        "error_flag": int(error_flag),
    }])

    pred = float(st.session_state["model"].predict(X_one)[0])
    st.metric("Prediksi flow", f"{pred:,.2f}")

    # Baseline (opsional) jika sudah dihitung
    if "baseline_test_pred" in st.session_state:
        # Baseline perlu data train; pakai baseline dari train_df dengan df dummy yang memiliki kolom relevan
        dummy = pd.DataFrame([{
            "detid": detid,
            "hour": int(hour),
            "dow": dow,
        }])
        # Hack: median_baseline butuh kolom TARGET untuk df_to_pred? tidak; cukup key kolom
        # Tapi fungsi kita merge; aman.
        base_pred = float(median_baseline(train_df, dummy)[0])
        st.write(f"Baseline median (detid, dow, hour): {base_pred:,.2f}")
