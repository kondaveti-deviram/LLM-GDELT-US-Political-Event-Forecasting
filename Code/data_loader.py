# data_loader.py

import pandas as pd
from config import DATA_PATH

def load_us_weekly_summaries():
    """
    Loads the GDELT CSV, filters to U.S. events, and returns
    a pd.Series indexed by Monday dates with two fields:
      - 'summary': the top-10 EventCode:Actor1Name (URL) strings
      - 'count'  : total number of filtered events that week
    """
    df = pd.read_csv(DATA_PATH, dtype={"Day": str})

    # Parse 'Day' → datetime
    sample = df["Day"].dropna().astype(str).head(5)
    if sample.str.match(r"^\d{8}$").all():
        fmt = "%Y%m%d"
    elif sample.str.match(r"^\d{4}-\d{2}-\d{2}$").all():
        fmt = "%Y-%m-%d"
    else:
        fmt = None

    df["Date"] = (
        pd.to_datetime(df["Day"], format=fmt, errors="coerce")
        if fmt else pd.to_datetime(df["Day"], errors="coerce")
    )
    df = df.dropna(subset=["Date"])

    # Filter to US
    df = df[df["ActionGeo_FullName"].str.contains("United States|USA", na=False)]
    df["EventCode"] = pd.to_numeric(df["EventCode"], errors="coerce")

    # Build text summaries
    df["CodeActorURL"] = (
        df["EventCode"].astype(int).astype(str)
        + ":"
        + df["Actor1Name"].fillna("UNKNOWN")
        + " ("
        + df["SOURCEURL"]
        + ")"
    )

    df = df.set_index("Date").sort_index()
    weekly_grp = df.groupby(pd.Grouper(freq="W-MON"))
    # summary = top-10 strings; count = total rows
    weekly = weekly_grp["CodeActorURL"].apply(lambda lst: "; ".join(lst.tolist()[:10]))
    counts = weekly_grp.size().rename("count")

    out = pd.DataFrame({"summary": weekly, "count": counts})
    return out
