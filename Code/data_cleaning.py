import pandas as pd

# File paths
input_file = "/Users/vaishnavimadhavaram/Downloads/event_forecast/us_political_events_last300days.csv"
output_file = "cleaned_us_political_events_300.csv"

# Columns needed
required_cols = [
    "GLOBALEVENTID", "SQLDATE",
    "Actor1Name", "Actor2Name",
    "EventCode", "QuadClass", "GoldsteinScale",
    "NumMentions", "NumSources", "NumArticles",
    "AvgTone",
    "ActionGeo_FullName", "ActionGeo_Lat", "ActionGeo_Long",
    "SOURCEURL"
]

# Prepare output file
first_chunk = True
total_rows_before = 0
total_rows_after = 0

chunk_size = 100_000
print(f"🔵 Starting chunked reading... (chunk size = {chunk_size})")

for chunk in pd.read_csv(input_file, usecols=required_cols, chunksize=chunk_size, low_memory=False):

    total_rows_before += len(chunk)

    # Rename columns
    chunk.rename(columns={
        "SQLDATE": "Day",
        "ActionGeo_Full_Name": "ActionGeo_Fullname"  # Correct naming if needed
    }, inplace=True)

    # Drop any row with missing important fields
    chunk.dropna(inplace=True)

    # Clean actors faster (vectorized)
    actor1_valid = chunk["Actor1Name"].astype(str).str.strip() != ""
    actor2_valid = chunk["Actor2Name"].astype(str).str.strip() != ""
    chunk = chunk[actor1_valid | actor2_valid]

    # Convert lat/lon safely
    chunk["ActionGeo_Lat"] = pd.to_numeric(chunk["ActionGeo_Lat"], errors="coerce")
    chunk["ActionGeo_Long"] = pd.to_numeric(chunk["ActionGeo_Long"], errors="coerce")
    chunk = chunk[
        (chunk["ActionGeo_Lat"].between(-90, 90)) &
        (chunk["ActionGeo_Long"].between(-180, 180))
    ]

    total_rows_after += len(chunk)

    # Save each chunk to disk
    if first_chunk:
        chunk.to_csv(output_file, index=False, mode='w')
        first_chunk = False
    else:
        chunk.to_csv(output_file, index=False, mode='a', header=False)

    print(f"✅ Processed chunk, kept {len(chunk)} rows.")

# Final report
print(f"\n✅ Finished! Kept {total_rows_after} of {total_rows_before} rows.")
