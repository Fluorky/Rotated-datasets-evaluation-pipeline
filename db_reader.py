import pandas as pd
import sqlite3

pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)
pd.set_option('display.max_colwidth', None)

df = pd.read_sql("SELECT * FROM training_logs", sqlite3.connect("mnist_logs.db"))
print(df)
