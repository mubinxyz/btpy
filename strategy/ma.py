import pandas as pd


df = pd.read_csv('../csv_data/csv_btcusd_5.csv', parse_dates=True, index_col=[0])

print(df.head())