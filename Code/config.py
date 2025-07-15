import os

# 1) Set your API key in the environment before running:

OPENAI_API_KEY = ""
DATA_PATH = "cleaned_us_political_events_pruned.csv" 

# Define domains by their EventRootCode roots and alert thresholds
DOMAINS = {
    'protest': {'roots': list(range(18,20)), 'alert_thresh': 0.6},
    'strike':  {'roots': [17],                'alert_thresh': 0.5},
    'attack':  {'roots': list(range(20,30)),  'alert_thresh': 0.4},
}

# How many bins to use for calibration plots
CALIBRATION_BINS = 5