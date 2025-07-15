# ingest_us_political_events.py

import os
from dotenv import load_dotenv
from google.cloud import bigquery
import pandas as pd

def fetch_us_political_events(days_back: int = 14) -> pd.DataFrame:
    load_dotenv()  
    client = bigquery.Client(project=os.getenv("BIGQUERY_PROJECT"))

    cnt = client.query(
    "SELECT COUNT(*) AS cnt "
    "FROM `gdelt-bq.gdeltv2.events` "
    "WHERE ActionGeo_CountryCode='US'"
    ).to_dataframe().iloc[0].cnt
    print("Total US events:", cnt)

    sql = f"""
    SELECT *
    FROM
      `gdelt-bq.gdeltv2.events`
    WHERE
      ActionGeo_CountryCode = 'US'                 
      AND CAST(SQLDATE AS STRING) BETWEEN
        FORMAT_DATE('%Y%m%d', DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY))
      AND
        FORMAT_DATE('%Y%m%d', CURRENT_DATE())
      AND 
        SAFE_CAST(EventRootCode AS INT64) BETWEEN 1 AND 19
    """

    df = client.query(sql).to_dataframe()
    return df

def main():
    df = fetch_us_political_events(days_back=300)
    print(f"Retrieved {len(df)} events.")
    df.to_csv("us_political_events_last150days.csv", index=False)
    print("Saved to us_political_events_last150days.csv")

if __name__ == "__main__":
    main()
