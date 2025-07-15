import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # Import os for file checking if needed elsewhere

from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    average_precision_score # Import average_precision_score
)
from sklearn.calibration import calibration_curve

from config import DOMAINS
# Make sure load_us_weekly_summaries is defined, possibly in data_loader.py
from data_loader import load_us_weekly_summaries 

plt.rcParams.update({"figure.autolayout": True})

# --- CONFIG ---
DOMAINS_LIST = list(DOMAINS.keys())
PRED_CSV_TPL = "predictions_expl_{}.csv" # Using the template from evaluate.py output
RATES_CSV = "domain_positive_rates.csv" # File for pre-calculated rates
CAL_BINS     = 5

# --- Helper functions with caching ---

# Cache loading summaries
@st.cache_data
def cached_load_summaries():
    print("Cache miss: Loading weekly summaries...") # Add print to see when cache misses
    return load_us_weekly_summaries()

# Cache loading prediction CSV for a specific domain
@st.cache_data
def load_prediction_data(domain):
    print(f"Cache miss: Loading prediction data for domain '{domain}'...") # Add print
    csv_path = PRED_CSV_TPL.format(domain)
    try:
        df = pd.read_csv(csv_path, parse_dates=["week"])
        if df.empty:
            st.warning(f"Predictions CSV '{csv_path}' is empty.")
            return None # Return None or empty df to indicate issue
        return df
    except FileNotFoundError:
        st.error(f"Missing prediction file: {csv_path}")
        st.stop() # Stop execution if prediction file is missing for selected domain
    except Exception as e:
        st.error(f"Error reading {csv_path}: {e}")
        st.stop()

# --- PAGE SETUP ---
st.set_page_config(page_title="Event Forecast Dashboard", layout="wide")
st.title("📊 GDELT-GPT Forecast Dashboard")
st.markdown("Explore forecasts for Protests, Strikes, and Attacks with metrics, distributions, and example cases.")

# --- SIDEBAR ---
domain = st.sidebar.selectbox("Select Domain", DOMAINS_LIST)
st.sidebar.markdown("Run `evaluate.py` to regenerate predictions and the domain rates summary file.")

# --- LOAD DATA USING CACHED FUNCTIONS ---
# weekly_summaries = cached_load_summaries() # Load only if needed later, otherwise skip
df = load_prediction_data(domain)

# Ensure df loaded correctly before proceeding
if df is None or df.empty:
    # load_prediction_data handles error messages and st.stop() or warnings
    st.stop() # Stop if df loading failed or returned empty

# --- COLUMN NAMES ---
# Define column names based on the CSV structure from evaluate.py
true_col     = "true"
raw_prob_col = "raw_prob"
# Use the prediction column used for the positive rate in evaluate.py (e.g., pred_05)
# Or if 'raw_pred' exists and is based on 0.5 threshold, use that. Let's assume 'pred_05' exists.
raw_pred_col = "raw_pred" # Or "raw_pred" if that's the column name from evaluate.py @ 0.5
cal_prob_col = "cal_prob"  if "cal_prob" in df.columns else None
cal_pred_col = "cal_pred"  if "cal_pred" in df.columns else None # Prediction based on calibrated prob

# Check if required columns exist
required_cols = [true_col, raw_prob_col, raw_pred_col, 'week', 'explanation']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
     st.error(f"Missing required columns in {PRED_CSV_TPL.format(domain)}: {', '.join(missing_cols)}")
     st.stop()

y_true = df[true_col]
y_prob = df[raw_prob_col]
y_pred = df[raw_pred_col] # Using prediction at 0.5 (or specified threshold)

# --- METRICS CARDS ---
try:
    brier = brier_score_loss(y_true, y_prob)
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auc   = roc_auc_score(y_true, y_prob) if y_true.nunique()>1 else np.nan
except Exception as e:
    st.error(f"Error calculating metrics: {e}")
    # Assign default values or stop
    brier, acc, prec, rec, f1, auc = [np.nan]*6

st.subheader("📈 Core Metrics (@ Thr=0.5)") # Specify threshold if appropriate
c1, c2, c3, c4 = st.columns(4)
c1.metric("Brier Score", f"{brier:.3f}" if not np.isnan(brier) else "N/A")
c2.metric("Accuracy",    f"{acc:.3f}" if not np.isnan(acc) else "N/A")
c3.metric("Precision",   f"{prec:.3f}" if not np.isnan(prec) else "N/A")
c4.metric("Recall",      f"{rec:.3f}" if not np.isnan(rec) else "N/A")
st.markdown(f"**F₁ Score:** {f1:.3f}" + (f"  |  **ROC-AUC:** {auc:.3f}" if not np.isnan(auc) else "") if not np.isnan(f1) else "")

st.markdown("---")

st.markdown("---")

# --- RELIABILITY DIAGRAM ---
st.subheader("🎯 Reliability Diagram")
try:
    # Ensure y_true and y_prob have the same length after potential drops
    # No drops here yet, should be aligned from start
    if len(y_true) != len(y_prob):
         raise ValueError("Length mismatch between y_true and y_prob for reliability plot.")

    pt_r, pp_r = calibration_curve(y_true, y_prob, n_bins=CAL_BINS, strategy="quantile")
    fig3, ax3 = plt.subplots(figsize=(3,2))
    ax3.plot(pp_r, pt_r, marker="s", markersize=4, label="Raw")
    if cal_prob_col:
        cal_probs_clean = pd.to_numeric(df[cal_prob_col], errors='coerce').dropna()
        # Align y_true with valid calibrated probabilities
        y_true_for_cal = y_true[cal_probs_clean.index] 
        if not cal_probs_clean.empty and len(y_true_for_cal) > 0 and y_true_for_cal.nunique() > 0:
             # Need at least two classes in the subset used for calibrated curve
             if y_true_for_cal.nunique() > 1 :
                 pt_c, pp_c = calibration_curve(y_true_for_cal, cal_probs_clean, n_bins=CAL_BINS, strategy="quantile")
                 ax3.plot(pp_c, pt_c, marker="o", markersize=4, label="Cal")
             else:
                  print("Note: Only one class present for calibrated reliability curve calculation.")
        else:
            print(f"Note: Not plotting calibrated curve due to empty/invalid data in '{cal_prob_col}' or index mismatch.")


    ax3.plot([0,1],[0,1],"k--", linewidth=0.8, label="Perfect") # Added label for clarity
    ax3.set_xlabel("Mean Predicted Prob", fontsize="small") # Updated label
    ax3.set_ylabel("Observed Frequency", fontsize="small") # Updated label
    ax3.legend(fontsize="small")
    ax3.tick_params(labelsize="small")
    st.pyplot(fig3)
except Exception as e:
    st.error(f"Error plotting reliability diagram: {e}")


st.markdown("---")

# --- ROC & PR & CM ---
# Add header common to all plots in this section
st.subheader("📉 Performance Curves & Matrix") 

if y_true.nunique() > 1:
    c1, c2 = st.columns(2) # Use columns for layout
    
    with c1: # Column 1 for PR Curve
        try:
            # Precision-Recall
            st.markdown("**Precision-Recall Curve**")
            precs, recs, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            fig5, ax5 = plt.subplots(figsize=(3,2)) # Adjusted size
            ax5.plot(recs, precs, linewidth=1.5, label=f"AP={ap:.2f}") # Slightly thicker line
            ax5.set_xlabel("Recall", fontsize="small")
            ax5.set_ylabel("Precision", fontsize="small")
            ax5.legend(fontsize="small")
            ax5.tick_params(labelsize="small")
            ax5.set_ylim(0, 1.05) # Ensure y-axis goes to 1
            ax5.set_xlim(0, 1.0)  # Ensure x-axis goes to 1
            st.pyplot(fig5)
        except Exception as e:
             st.error(f"Error plotting PR curve: {e}")

    with c2: # Column 2 for Confusion Matrix
        try:
            # Confusion Matrix
            st.markdown("**Confusion Matrix (@ Thr=0.5)**") # Clarify threshold if needed
            cm = confusion_matrix(y_true, y_pred)
            fig6, ax6 = plt.subplots(figsize=(2.5,2)) # Adjusted size
            im = ax6.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1) # Handle cm.max=0 case
            # Determine text color based on background
            thresh = cm.max() / 2. if cm.max() > 0 else 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax6.text(j, i, f"{cm[i,j]:,}", # Format number with comma
                             ha="center", va="center",
                             color="white" if cm[i,j] > thresh else "black",
                             fontsize="medium") # Slightly larger font
            ax6.set_xticks(np.arange(cm.shape[1])); ax6.set_xticklabels([f"Pred={i}" for i in range(cm.shape[1])], fontsize="small")
            ax6.set_yticks(np.arange(cm.shape[0])); ax6.set_yticklabels([f"True={i}" for i in range(cm.shape[0])], fontsize="small")
            ax6.set_xlabel("Predicted Label", fontsize="small"); ax6.set_ylabel("True Label", fontsize="small")
            plt.setp(ax6.get_xticklabels(), rotation=0, ha="center") # Ensure labels are horizontal
            st.pyplot(fig6)
        except Exception as e:
            st.error(f"Error plotting confusion matrix: {e}")
else:
     st.info("ROC curve, PR curve, and Confusion Matrix require both classes (0 and 1) to be present in the true labels.")



# --- PROMPT & MODEL RATIONALE ---
st.markdown("---")
st.subheader("💡 Prompt & Model Rationale")
st.markdown(
    "Select a week to see the exact weekly summary (prompt) sent to the LLM, the "
    "model’s predicted probability, and the key events it cited."
)

week_sel = st.selectbox(
    "Week to inspect",
    df["week"].dt.date.astype(str).tolist()
)
row = df[df["week"].dt.date.astype(str) == week_sel].iloc[0]

with st.expander(f"🔍 Weekly Summary for {week_sel}"):
    for ev in str(row["summary"]).split("; "):
        st.markdown(f"- {ev}", unsafe_allow_html=False)

st.markdown(f"**Predicted Probability:** {row[raw_prob_col]:.2%}")

st.markdown("**Key Evidence (why):**")
for snippet in str(row["explanation"]).split("; "):
    st.markdown(f"- {snippet}")



# --- DOWNLOAD ---
st.markdown("---")
try:
    # Prepare data for download (e.g., convert week back to string)
    df_download = df.copy()
    df_download['week'] = df_download['week'].dt.strftime('%Y-%m-%d')
    
    csv_data = df_download.to_csv(index=False).encode("utf-8")
    
    st.download_button(
        "⬇️ Download Predictions CSV",
        data=csv_data,
        file_name=PRED_CSV_TPL.format(domain), # Use template for filename
        mime="text/csv"
    )
except Exception as e:
     st.error(f"Error preparing download link: {e}")