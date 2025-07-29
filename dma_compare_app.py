import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

st.set_page_config(layout="wide")
st.title("DMA Curve Comparison (% Difference Analyzer)")

# === Upload Inputs ===
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Excel file for Curve A", type=["xlsx"])
    label_a = st.text_input("Label for Curve A", value="Sample A")
with col2:
    file_b = st.file_uploader("Upload Excel file for Curve B", type=["xlsx"])
    label_b = st.text_input("Label for Curve B", value="Sample B")

smooth = st.checkbox("Apply smoothing to curves?", value=True)

# === Process Once Both Files Are Uploaded ===
if file_a and file_b:
    try:
        # Load and extract
        def load_curve(file):
            df = pd.read_excel(file, header=None)
            df = df.dropna(subset=[0, 1, 2, 3])  # drop rows with NaNs in any required column
            for col in [0, 1, 2, 3]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df.dropna(subset=[0, 1, 2, 3])
            
            freq = df.iloc[:, 0].values
            storage = df.iloc[:, 1].values
            loss = df.iloc[:, 2].values
            tand = df.iloc[:, 3].values
            return np.log10(freq), storage, loss, tand

        log_freq_a, stor_a, loss_a, tan_a = load_curve(file_a)
        log_freq_b, stor_b, loss_b, tan_b = load_curve(file_b)

        # Remove duplicates
        def dedup_all(log_freq, storage, loss, tand):
            df = pd.DataFrame({
                "log_freq": log_freq,
                "storage": storage,
                "loss": loss,
                "tand": tand
            })
            df = df.groupby("log_freq", as_index=False).mean()
            return (
                df["log_freq"].values,
                df["storage"].values,
                df["loss"].values,
                df["tand"].values
            )
            
        log_freq_a, stor_a, loss_a, tan_a = dedup_all(log_freq_a, stor_a, loss_a, tan_a)
        log_freq_b, stor_b, loss_b, tan_b = dedup_all(log_freq_b, stor_b, loss_b, tan_b)



        # Common frequency range
        min_common = max(min(log_freq_a), min(log_freq_b))
        max_common = min(max(log_freq_a), max(log_freq_b))
        common_log_freq = np.linspace(min_common, max_common, 500)

        def interpolate_and_smooth(x, y):
            f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
            y_interp = f(common_log_freq)
            if smooth:
                y_interp = savgol_filter(y_interp, 11, 3, mode='interp')
            return y_interp

        # Interpolated curves
        stor_a_i = interpolate_and_smooth(log_freq_a, stor_a)
        stor_b_i = interpolate_and_smooth(log_freq_b, stor_b)
        loss_a_i = interpolate_and_smooth(log_freq_a, loss_a)
        loss_b_i = interpolate_and_smooth(log_freq_b, loss_b)
        tan_a_i = interpolate_and_smooth(log_freq_a, tan_a)
        tan_b_i = interpolate_and_smooth(log_freq_b, tan_b)

        # Percent difference
        def percent_diff(a, b):
            return 100 * (a - b) / b

        freq_display = 10**common_log_freq
        plots = {
            "Storage Modulus (Pa)": (stor_a_i, stor_b_i),
            "Loss Modulus (Pa)": (loss_a_i, loss_b_i),
            "Tan Delta": (tan_a_i, tan_b_i)
        }

        for title, (curve_a, curve_b) in plots.items():
            delta = percent_diff(curve_a, curve_b)
            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax1.set_title(f"{title} Comparison")
            ax1.plot(freq_display, curve_a, label=label_a, color='blue')
            ax1.plot(freq_display, curve_b, label=label_b, color='red')
            ax1.set_xscale('log')
            ax1.set_yscale('log' if title != "Tan Delta" else 'linear')
            ax1.set_ylabel(title)
            ax1.set_xlabel("Frequency (Hz)")
            ax1.legend()
            ax1.grid(True)

            # % diff overlay
            ax2 = ax1.twinx()
            ax2.plot(freq_display, delta, color='purple', linestyle='--', label="% Difference")
            ax2.set_ylabel("% Difference")
            ax2.legend(loc='lower right')

            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error while processing files: {e}")
