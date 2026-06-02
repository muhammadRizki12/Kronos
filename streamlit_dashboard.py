from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GROUND_TRUTH_DIR = DATA_DIR / "ground-truth"
FORECAST_DIR = DATA_DIR / "prediction"

PRICE_COLS = ["open", "high", "low", "close"]
CLOSE_COL = "close"

st.set_page_config(
    page_title="EUR/IDR Forecast vs Ground Truth",
    page_icon="📈",
    layout="wide",
)

st.title("EUR/IDR Dashboard")
st.caption("Dashboard fokus metrik close: ringkasan, rentang tanggal, perbandingan forecast, dan detail data.")


def discover_csv_files(folder: Path):
    if not folder.exists():
        return []
    return sorted([path for path in folder.glob("*.csv") if path.is_file()])


@st.cache_data(show_spinner=False)
def load_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [str(column).strip().lower() for column in df.columns]

    rename_map = {}
    if "unnamed: 0" in df.columns:
        rename_map["unnamed: 0"] = "timestamp"
    if "date" in df.columns and "timestamp" not in df.columns:
        rename_map["date"] = "timestamp"
    if "timestamps" in df.columns and "timestamp" not in df.columns:
        rename_map["timestamps"] = "timestamp"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "timestamp" not in df.columns:
        first_column = df.columns[0]
        if first_column not in PRICE_COLS:
            df = df.rename(columns={first_column: "timestamp"})

    if "timestamp" not in df.columns:
        raise ValueError(f"Kolom timestamp tidak ditemukan di file {file_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for column in PRICE_COLS + ["volume", "amount"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = np.nan

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def merge_forecast_and_gt(forecast_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
    merged = forecast_df[["timestamp"] + [c for c in PRICE_COLS if c in forecast_df.columns]].merge(
        gt_df[["timestamp"] + [c for c in PRICE_COLS if c in gt_df.columns]],
        on="timestamp",
        how="left",
        suffixes=("_forecast", "_gt"),
    )

    for column in PRICE_COLS:
        forecast_col = f"{column}_forecast"
        gt_col = f"{column}_gt"
        if forecast_col in merged.columns and gt_col in merged.columns:
            merged[f"{column}_error"] = merged[forecast_col] - merged[gt_col]
            merged[f"{column}_abs_error"] = merged[f"{column}_error"].abs()
            merged[f"{column}_pct_error"] = np.where(
                merged[gt_col].abs() > 0,
                merged[f"{column}_abs_error"] / merged[gt_col].abs() * 100,
                np.nan,
            )
    return merged.sort_values("timestamp").reset_index(drop=True)


def calc_metrics(merged_df: pd.DataFrame, price_col: str) -> dict:
    forecast_col = f"{price_col}_forecast"
    gt_col = f"{price_col}_gt"

    if forecast_col not in merged_df.columns or gt_col not in merged_df.columns:
        return {
            "points": len(merged_df),
            "overlap": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "bias": np.nan,
            "directional_accuracy": np.nan,
            "last_gap": np.nan,
        }

    overlap = merged_df.loc[merged_df[forecast_col].notna() & merged_df[gt_col].notna()].copy()

    if overlap.empty:
        return {
            "points": 0,
            "overlap": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "bias": np.nan,
            "directional_accuracy": np.nan,
            "last_gap": np.nan,
        }

    error = overlap[forecast_col] - overlap[gt_col]
    abs_error = error.abs()
    gt_diff = overlap[gt_col].diff()
    forecast_diff = overlap[forecast_col].diff()
    direction_mask = gt_diff.notna() & forecast_diff.notna()
    directional_accuracy = np.nan
    if direction_mask.any():
        directional_accuracy = (
            (np.sign(gt_diff[direction_mask]) == np.sign(forecast_diff[direction_mask])).mean() * 100
        )

    return {
        "points": len(merged_df),
        "overlap": len(overlap),
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "mape": float((abs_error / overlap[gt_col].abs()).replace([np.inf, -np.inf], np.nan).mean() * 100),
        "bias": float(error.mean()),
        "directional_accuracy": float(directional_accuracy) if pd.notna(directional_accuracy) else np.nan,
        "last_gap": float(abs_error.iloc[-1]),
    }


def calc_metrics_for_close(merged_df: pd.DataFrame) -> dict:
    return calc_metrics(merged_df, CLOSE_COL)


def format_metric(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.4f}"


def metric_cards_close(metrics: dict, label: str) -> None:
    card_cols = st.columns(4)
    with card_cols[0]:
        st.metric(f"MAE ({label})", format_metric(metrics["mae"]))
    with card_cols[1]:
        st.metric(f"RMSE ({label})", format_metric(metrics["rmse"]))
    with card_cols[2]:
        mape_value = "-" if pd.isna(metrics["mape"]) else f"{metrics['mape']:.2f}%"
        st.metric(f"MAPE ({label})", mape_value)
    with card_cols[3]:
        st.metric(f"Directional Acc. ({label})", "-" if pd.isna(metrics["directional_accuracy"]) else f"{metrics['directional_accuracy']:.2f}%")


forecast_files = discover_csv_files(FORECAST_DIR)
gt_files = discover_csv_files(GROUND_TRUTH_DIR)

if not forecast_files:
    st.error(f"Tidak ada file forecast CSV di {FORECAST_DIR}")
    st.stop()

if not gt_files:
    st.error(f"Tidak ada file ground truth CSV di {GROUND_TRUTH_DIR}")
    st.stop()

with st.sidebar:
    st.header("Pengaturan")
    forecast_path = st.selectbox(
        "Pilih forecast utama",
        options=forecast_files,
        format_func=lambda path: path.name,
    )
    extra_forecasts = st.multiselect(
        "Tambah forecast lain (opsional)",
        options=[path for path in forecast_files if path != forecast_path],
        format_func=lambda path: path.name,
    )
    gt_path = st.selectbox(
        "Pilih file ground truth",
        options=gt_files,
        format_func=lambda path: path.name,
        index=0,
    )
    show_raw_table = st.checkbox("Tampilkan semua baris detail close", value=False)

selected_forecasts = [forecast_path] + [path for path in extra_forecasts if path != forecast_path]

gt_df = load_csv(str(gt_path))
main_forecast_df = load_csv(str(forecast_path))
main_merged_df = merge_forecast_and_gt(main_forecast_df, gt_df)

start_date = main_merged_df["timestamp"].min().date()
end_date = main_merged_df["timestamp"].max().date()
selected_range = st.slider(
    "Rentang tanggal",
    min_value=start_date,
    max_value=end_date,
    value=(start_date, end_date),
)

filtered_main = main_merged_df[
    (main_merged_df["timestamp"].dt.date >= selected_range[0])
    & (main_merged_df["timestamp"].dt.date <= selected_range[1])
].copy()
filtered_main = cast(pd.DataFrame, filtered_main)

forecast_col = f"{CLOSE_COL}_forecast"
gt_col = f"{CLOSE_COL}_gt"
error_col = f"{CLOSE_COL}_error"
abs_error_col = f"{CLOSE_COL}_abs_error"
pct_error_col = f"{CLOSE_COL}_pct_error"

missing_columns = [column for column in [forecast_col, gt_col, error_col, abs_error_col, pct_error_col] if column not in main_merged_df.columns]
if missing_columns:
    st.warning(
        "Beberapa kolom yang dibutuhkan tidak ada di hasil merge: " + ", ".join(missing_columns) + ". "
        "Dashboard tetap ditampilkan, tetapi metrik/grafik untuk harga yang hilang akan kosong."
    )

st.subheader("1) Ringkasan Metrics")
main_metrics = calc_metrics_for_close(filtered_main)
metric_cards_close(main_metrics, forecast_path.stem)

st.subheader("2) Rentang Tanggal")
st.write(f"Periode aktif: **{selected_range[0]}** s/d **{selected_range[1]}**")

comparison_rows = []
close_series_df = pd.DataFrame({"timestamp": gt_df["timestamp"]})
close_series_df = close_series_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
gt_close_df = pd.DataFrame({"timestamp": gt_df["timestamp"], "close_gt": gt_df[CLOSE_COL]})
close_series_df = close_series_df.merge(gt_close_df, on="timestamp", how="left")

for selected_forecast in selected_forecasts:
    selected_forecast_df = load_csv(str(selected_forecast))
    selected_merged_df = merge_forecast_and_gt(selected_forecast_df, gt_df)
    selected_filtered = selected_merged_df[
        (selected_merged_df["timestamp"].dt.date >= selected_range[0])
        & (selected_merged_df["timestamp"].dt.date <= selected_range[1])
    ].copy()
    selected_filtered = cast(pd.DataFrame, selected_filtered)

    selected_metrics = calc_metrics_for_close(selected_filtered)
    comparison_rows.append(
        {
            "forecast": selected_forecast.name,
            "points": selected_metrics["points"],
            "overlap": selected_metrics["overlap"],
            "mae": selected_metrics["mae"],
            "rmse": selected_metrics["rmse"],
            "mape": selected_metrics["mape"],
            "bias": selected_metrics["bias"],
            "directional_accuracy": selected_metrics["directional_accuracy"],
            "last_gap": selected_metrics["last_gap"],
        }
    )

    selected_close_df = pd.DataFrame(
        {
            "timestamp": selected_forecast_df["timestamp"],
            f"close_forecast_{selected_forecast.stem}": selected_forecast_df[CLOSE_COL],
        }
    )
    close_series_df = close_series_df.merge(selected_close_df, on="timestamp", how="left")

comparison_df = pd.DataFrame(comparison_rows)

st.subheader("3) Perbandingan Metrics")
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

chart_left, chart_right = st.columns([2, 1])

with chart_left:
    st.subheader("Perbandingan Close")
    fig = go.Figure()
    filtered_series = close_series_df[
        (close_series_df["timestamp"].dt.date >= selected_range[0])
        & (close_series_df["timestamp"].dt.date <= selected_range[1])
    ].copy()

    if "close_gt" in filtered_series.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_series["timestamp"],
                y=filtered_series["close_gt"],
                name="Ground Truth",
                mode="lines",
                line=dict(color="#2563eb", width=3),
            )
        )
    forecast_color_palette = ["#f97316", "#0f766e", "#7c3aed", "#dc2626", "#ca8a04", "#0891b2"]
    for idx, selected_forecast in enumerate(selected_forecasts):
        forecast_close_col = f"close_forecast_{selected_forecast.stem}"
        if forecast_close_col in filtered_series.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered_series["timestamp"],
                    y=filtered_series[forecast_close_col],
                    name=f"Forecast: {selected_forecast.stem}",
                    mode="lines",
                    line=dict(color=forecast_color_palette[idx % len(forecast_color_palette)], width=2),
                )
            )

    if len(fig.data) > 0:
        fig.update_layout(
            template="plotly_white",
            height=520,
            hovermode="x unified",
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h"),
        )
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Data close tidak tersedia untuk visualisasi perbandingan.")



st.subheader("4) Detail Data (Close)")
close_detail_cols = ["timestamp", "close_gt"] + [
    column for column in close_series_df.columns if column.startswith("close_forecast_")
]
detail_df = close_series_df[close_detail_cols].copy()
detail_df = detail_df[
    (detail_df["timestamp"].dt.date >= selected_range[0])
    & (detail_df["timestamp"].dt.date <= selected_range[1])
].sort_values("timestamp")

if show_raw_table:
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
else:
    st.dataframe(detail_df.head(30), use_container_width=True, hide_index=True)

st.download_button(
    label="Download hasil merge CSV",
    data=detail_df.to_csv(index=False).encode("utf-8"),
    file_name=f"close_comparison_{gt_path.stem}.csv",
    mime="text/csv",
)
