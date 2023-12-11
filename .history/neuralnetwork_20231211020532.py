import pandas as pd

file_path = './data.xlsx'

try:
    data = pd.read_excel(file_path)
    preview = data.head()
    print(preview)
except Exception as e:
    print(e)