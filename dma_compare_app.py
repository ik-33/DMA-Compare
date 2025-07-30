import streamlit as st
import pandas as pd
import numpy as np
import io
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
        
        #excel file download
        def create_excel_download(
            common_freq,
            smoothed_storage_A, smoothed_loss_A, smoothed_tan_A,
            smoothed_storage_B, smoothed_loss_B, smoothed_tan_B,
            curve_name_A, curve_name_B
        ):
            # Compute % difference: (B - A) / A * 100
            percent_diff_storage = (smoothed_storage_B - smoothed_storage_A) / smoothed_storage_A * 100
            percent_diff_loss = (smoothed_loss_B - smoothed_loss_A) / smoothed_loss_A * 100
            percent_diff_tan = (smoothed_tan_B - smoothed_tan_A) / smoothed_tan_A * 100
            
            #undo log(frequency)
            common_freq_hz = 10**common_freq
            
            #Full smoothed curve & % diff values
            df_out = pd.DataFrame({
                "Frequency (Hz)": common_freq_hz,

                f"{curve_name_A} Storage Modulus": smoothed_storage_A,
                f"{curve_name_B} Storage Modulus": smoothed_storage_B,
                "% Diff Storage Modulus": percent_diff_storage,

                f"{curve_name_A} Loss Modulus": smoothed_loss_A,
                f"{curve_name_B} Loss Modulus": smoothed_loss_B,
                "% Diff Loss Modulus": percent_diff_loss,

                f"{curve_name_A} Tan Delta": smoothed_tan_A,
                f"{curve_name_B} Tan Delta": smoothed_tan_B,
                "% Diff Tan Delta": percent_diff_tan,
            })
            
            # Summary at specific frequencies
            target_freqs = np.array([1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7])

            interp_storage = interp1d(common_freq_hz, percent_diff_storage, kind='linear', bounds_error=False, fill_value=np.nan)
            interp_loss = interp1d(common_freq_hz, percent_diff_loss, kind='linear', bounds_error=False, fill_value=np.nan)
            interp_tan = interp1d(common_freq_hz, percent_diff_tan, kind='linear', bounds_error=False, fill_value=np.nan)

            df_summary = pd.DataFrame({
                "Frequency (Hz)": target_freqs,
                "% Diff Storage Modulus": interp_storage(target_freqs),
                "% Diff Loss Modulus": interp_loss(target_freqs),
                "% Diff Tan Delta": interp_tan(target_freqs)
            })

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_out.to_excel(writer, index=False, sheet_name='TTS Comparison')
                df_summary.to_excel(writer, index=False, sheet_name='Summary % Difference')

            output.seek(0)
            return output

        # call excel download function
        excel_file = create_excel_download(
            common_log_freq,
            stor_a_i, loss_a_i, tan_a_i,
            stor_b_i, loss_b_i, tan_b_i,
            label_a, label_b
        )

        # Download Button
        st.download_button(
            label = "Download Smoothed & % Difference Data (Excel)",
            data = excel_file,
            file_name = f"{label_a}_vs_{label_b}_comparison.xlsx",
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

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


# to run in command line type:
# streamlit run dma_compare_app.py