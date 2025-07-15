# validate.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

DOMAINS       = ["protest","strike","attack"]
PRED_CSV_TPL  = "predictions_expl_{}.csv"
MIN_TRAIN_WEEKS = 5   # start evaluation after this many weeks

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute ECE: weighted avg abs difference between bin acc & bin pred."""
    bins = np.linspace(0,1,n_bins+1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids==b
        if mask.sum()==0: continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum()/len(y_true) * abs(acc-conf)
    return ece

def evaluate_domain(domain):
    df = pd.read_csv(PRED_CSV_TPL.format(domain), parse_dates=["week"])
    df = df.sort_values("week").reset_index(drop=True)
    N = len(df)

    # collect per-fold metrics
    metrics = {
        "brier_raw":[],"brier_cal":[],
        "auc_raw":[],  "auc_cal":[],
        "acc_raw":[],  "acc_cal":[],
        "prec_raw":[], "prec_cal":[],
        "rec_raw":[],  "rec_cal":[],
        "f1_raw":[],   "f1_cal":[],
        "ece_raw":[],  "ece_cal":[]  
    }

    # reliability curves aggregated
    all_y_true, all_p_raw, all_p_cal = [], [], []

    for t in tqdm(range(MIN_TRAIN_WEEKS, N)):
        train = df.iloc[:t]
        test  = df.iloc[t:t+1]

        y_tr, p_tr = train["true"].values, train["raw_prob"].values
        y_te, p_te = test["true"].values,  test["raw_prob"].values

        # Raw metrics on test
        metrics["brier_raw"].append(brier_score_loss(y_te, p_te))
        metrics["auc_raw"].append(roc_auc_score(y_te, p_te) if len(np.unique(y_tr))>1 else np.nan)
        pred_raw = (p_te>=0.5).astype(int)
        metrics["acc_raw"].append(accuracy_score(y_te, pred_raw))
        metrics["prec_raw"].append(precision_score(y_te, pred_raw, zero_division=0))
        metrics["rec_raw"].append(recall_score(y_te, pred_raw, zero_division=0))
        metrics["f1_raw"].append(f1_score(y_te, pred_raw, zero_division=0))
        metrics["ece_raw"].append(expected_calibration_error(y_te, p_te))

        # Fit Platt calibrator on train
        if len(np.unique(y_tr))>1:
            lr = LogisticRegression(solver="lbfgs")
            lr.fit(p_tr.reshape(-1,1), y_tr)
            p_te_cal = lr.predict_proba(p_te.reshape(-1,1))[:,1]
            metrics["brier_cal"].append(brier_score_loss(y_te, p_te_cal))
            metrics["auc_cal"].append(roc_auc_score(y_te, p_te_cal))
            pred_cal = (p_te_cal>=0.5).astype(int)
            metrics["acc_cal"].append(accuracy_score(y_te, pred_cal))
            metrics["prec_cal"].append(precision_score(y_te, pred_cal, zero_division=0))
            metrics["rec_cal"].append(recall_score(y_te, pred_cal, zero_division=0))
            metrics["f1_cal"].append(f1_score(y_te, pred_cal, zero_division=0))
            metrics["ece_cal"].append(expected_calibration_error(y_te, p_te_cal))
            all_p_cal.append(p_te_cal[0])
        else:
            # no calibration possible
            for k in ["brier_cal","auc_cal","acc_cal","prec_cal","rec_cal","f1_cal","ece_cal"]:
                metrics[k].append(np.nan)
            all_p_cal.append(p_te[0])

        all_y_true.append(y_te[0])
        all_p_raw.append(p_te[0])

    return df, metrics

def summarize(metrics):
    out = {}
    for k,vals in metrics.items():
        arr = np.array(vals)[~np.isnan(vals)]
        mean = arr.mean() if len(arr)>0 else np.nan
        lo, hi = np.percentile(arr, [2.5,97.5]) if len(arr)>1 else (np.nan, np.nan)
        out[k] = (mean, lo, hi)
    return out

if __name__=="__main__":
    for domain in DOMAINS:
        print(f"\n=== VALIDATION: {domain} ===")
        df, mets = evaluate_domain(domain)
        summary = summarize(mets)
        for k,(m,lo,hi) in summary.items():
            print(f"{k:10s} → {m:.3f}  [{lo:.3f},{hi:.3f}]")

        # plot reliability across all folds
        fig, ax = plt.subplots()
        pt_raw, pp_raw = calibration_curve(df["true"].iloc[MIN_TRAIN_WEEKS:], 
                                           mets["brier_raw"][:len(df)-MIN_TRAIN_WEEKS], 
                                           n_bins=10, strategy='uniform')
        ax.plot(pp_raw, pt_raw, marker="s", label="Raw")
        if not all(np.isnan(mets["brier_cal"])):
            pt_cal, pp_cal = calibration_curve(df["true"].iloc[MIN_TRAIN_WEEKS:], 
                                               mets["brier_cal"][:len(df)-MIN_TRAIN_WEEKS], 
                                               n_bins=10, strategy='uniform')
            ax.plot(pp_cal, pt_cal, marker="o", label="Calibrated")
        ax.plot([0,1],[0,1],"k--")
        ax.set_title(f"{domain} reliability (fold‐by‐fold)")
        ax.set_xlabel("Mean Pred Prob")
        ax.set_ylabel("Observed Frac")
        plt.show()
