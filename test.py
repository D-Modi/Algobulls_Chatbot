import pandas as pd
from datetime import datetime

# Sample data with mixed datetime formats
data = ["23-04-2024 14:34", "04/23/2024 2:34", "2024-04-23 14:34"]
df = pd.DataFrame(data, columns=['datetime'])

# Define parsing function
def parse_datetime(date_str):
    formats = [
        "%d-%m-%Y %H:%M",
        "%m/%d/%Y %I:%M",
        "%Y-%m-%d %H:%M"
    ]
    for fmt in formats:
        try:
            x = (datetime.strptime(date_str, fmt))
            print(type((x)))
            return x
        except ValueError:
            continue
    raise ValueError(f"Date format not recognized: {date_str}")

# Apply parsing function
df['datetime'].apply(parse_datetime)

# Format datetime to the desired format
print(df)
