# evaluate.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from forecast import forecast_summary
from config import DATA_PATH, DOMAINS
from data_loader import load_us_weekly_summaries


def load_ground_truth_for_roots(roots, start, end):
    """
    Build a weekly series (Mondays) from start_date to end_date:
      1 if any EventRootCode in `roots` occurred that week in the US, else 0.
    """
    df = pd.read_csv(DATA_PATH, dtype={'Day': str})
    # 1) Parse dates
    sample = df['Day'].dropna().astype(str).head(5)
    if sample.str.match(r'^\d{8}$').all():
        fmt = '%Y%m%d'
    elif sample.str.match(r'^\d{4}-\d{2}-\d{2}$').all():
        fmt = '%Y-%m-%d'
    else:
        fmt = None

    df['Date'] = (
        pd.to_datetime(df['Day'], format=fmt, errors='coerce')
        if fmt else pd.to_datetime(df['Day'], errors='coerce')
    )
    df = df.dropna(subset=['Date'])
    # 2) Filter to U.S.
    df = df[df['ActionGeo_FullName'].str.contains('United States|USA', na=False)]
    # 3) Compute root code flag
    df['EventCode']     = pd.to_numeric(df['EventCode'], errors='coerce')
    df['EventRootCode'] = (df['EventCode'] // 10).astype(int)
    df['is_target']     = df['EventRootCode'].isin(roots).astype(int)
    # 4) Weekly max
    weekly_raw = (
        df.set_index('Date')
          .groupby(pd.Grouper(freq='W-MON'))['is_target']
          .max()
    )
    # 5) Reindex onto all Mondays
    all_weeks = pd.date_range(start=start, end=end, freq='W-MON')
    return weekly_raw.reindex(all_weeks, fill_value=0)


def find_best_threshold(y_true, y_prob):
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0, 1, 101):
        preds = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def evaluate():
    # 1) Load weekly summaries
    summaries = load_us_weekly_summaries()
    start, end = summaries.index.min(), summaries.index.max()

    for domain, cfg in DOMAINS.items():
        roots, alert_th = cfg['roots'], cfg['alert_thresh']
        print(f"\n=== Domain: {domain} (roots={roots}) ===")

        # 2) Ground truth over full range
        truth = load_ground_truth_for_roots(roots, start, end)

        # 3) Build a full-week table with summary & count
        weekly_full = summaries.reindex(
            truth.index,
            fill_value={'summary': "", 'count': 0}
        )

        # 4) Forecast loop
        records = []
        for wk, row in weekly_full.iterrows():
            summary = row['summary']
            count   = row['count']
            prob, rationale = forecast_summary(summary, count, domain)

            true = int(truth.loc[wk])
            if prob is None:
                continue

            records.append({
                'week': wk.date(),
                'raw_prob': prob,
                'summary': summary,
                'explanation': rationale,
                'true': true
            })

        df = pd.DataFrame(records)
        if df.empty:
            print("  No valid forecasts—skipping.")
            continue

        # 5) Class balance
        print("  Label counts (true):", df['true'].value_counts().to_dict())

        # 6) Raw prediction at 0.5
        df['raw_pred'] = (df['raw_prob'] >= 0.5).astype(int)

        # 7) Raw metrics
        b_raw = brier_score_loss(df['true'], df['raw_prob'])
        acc   = accuracy_score(df['true'], df['raw_pred'])
        prec  = precision_score(df['true'], df['raw_pred'], zero_division=0)
        rec   = recall_score(df['true'], df['raw_pred'], zero_division=0)
        f1    = f1_score(df['true'], df['raw_pred'], zero_division=0)
        auc   = (roc_auc_score(df['true'], df['raw_prob'])
                 if df['true'].nunique() > 1 else float('nan'))
        print(f"  Raw → Brier: {b_raw:.3f}, Acc: {acc:.3f}, "
              f"Prec: {prec:.3f}, Rec: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

        # 8) Threshold tuning
        best_thr, best_f1 = find_best_threshold(df['true'], df['raw_prob'])
        df['tuned_pred'] = (df['raw_prob'] >= best_thr).astype(int)
        acc_tuned = accuracy_score(df['true'], df['tuned_pred'])
        # print(f"  Best F1={best_f1:.3f} at thr={best_thr:.2f}, Tuned Acc: {acc_tuned:.3f}")

        # 9) Alert
        next_p = df.iloc[-1]['raw_prob']
        if next_p >= alert_th:
            print(f"  ALERT: next-week raw_prob={next_p:.2f} ≥ alert_thresh={alert_th}")

        # 10) Calibration
        if df['true'].nunique() > 1:
            lr = LogisticRegression(solver='lbfgs')
            lr.fit(df[['raw_prob']], df['true'])
            df['cal_prob'] = lr.predict_proba(df[['raw_prob']])[:,1]
            df['cal_pred'] = (df['cal_prob'] >= 0.5).astype(int)
            b_cal = brier_score_loss(df['true'], df['cal_prob'])
            a_cal = accuracy_score(df['true'], df['cal_pred'])
            #print(f"  Calibrated → Brier: {b_cal:.3f}, Acc: {a_cal:.3f}")
        else:
            print("  Only one class—skipping calibration.")
            df['cal_prob'], df['cal_pred'] = np.nan, np.nan

        # 11) Save predictions with explanations
        out = f"predictions_expl_{domain}.csv"
        df.to_csv(out, index=False)
        print(f"  Saved → {out}")

        # 12) Plot calibration
        fig, ax = plt.subplots()
        pt, pp = calibration_curve(df['true'], df['raw_prob'], n_bins=10, strategy='uniform')
        ax.plot(pp, pt, marker='s', label='Raw')
        if df['cal_prob'].notna().any():
            pt2, pp2 = calibration_curve(df['true'], df['cal_prob'], n_bins=10, strategy='uniform')
            ax.plot(pp2, pt2, marker='o', label='Calibrated')
        ax.plot([0,1], [0,1], 'k--')
        ax.set_title(f"{domain.capitalize()} Calibration")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluate()
