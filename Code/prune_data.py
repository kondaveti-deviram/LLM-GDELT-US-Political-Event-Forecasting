# prune_weeks.py

import pandas as pd

# --- CONFIGURE ---

INPUT_CSV  = "cleaned_us_political_events_300.csv"
OUTPUT_CSV = "cleaned_us_political_events_pruned.csv"

# Domain → EventRootCodes (must match config.DOMAIN roots)
DOMAIN_ROOTS = {
    "protest": [18, 19],
    "strike":  [17],
    "attack":  list(range(20,30)),
}

# Weeks to prune per domain (YYYY-MM-DD = the Monday of the week)
PRUNE_WEEKS = {
    "protest": ["2024-12-10", "2025-01-14"],
    "strike":  ["2025-02-04", "2025-03-18"],
    "attack":  ["2024-12-17", "2025-04-15"],
}

def main():
    # 1) Load & parse
    df = pd.read_csv(INPUT_CSV, dtype={"Day": str})
    df["Date"]   = pd.to_datetime(df["Day"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"])

    # 2) Compute the Monday of each event's week
    df["WeekMon"] = df["Date"].dt.to_period("W-MON").apply(lambda p: p.start_time.date().isoformat())

    # 3) Extract root code
    df["EventCode"]     = pd.to_numeric(df["EventCode"], errors="coerce")
    df["EventRootCode"] = (df["EventCode"] // 10).astype(int)

    # 4) For each domain, drop exactly those rows
    mask = pd.Series(True, index=df.index)
    for domain, weeks in PRUNE_WEEKS.items():
        roots = set(DOMAIN_ROOTS[domain])
        for wk in weeks:
            # rows to drop: week == wk AND root in this domain
            to_drop = (df["WeekMon"] == wk) & (df["EventRootCode"].isin(roots))
            print(f"Dropping {to_drop.sum()} rows for domain={domain}, week={wk}")
            mask &= ~to_drop

    # 5) Apply and save
    df_pruned = df.loc[mask].drop(columns=["WeekMon"])
    df_pruned.to_csv(OUTPUT_CSV, index=False)
    kept, dropped = len(df_pruned), len(df) - len(df_pruned)
    print(f"\nFinal: Kept {kept}, Dropped {dropped} rows → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
