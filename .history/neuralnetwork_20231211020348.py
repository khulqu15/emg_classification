import pandas as pd

file_path = './data.xlsx'

try:
    data = pd.read_excel(file_path)
    preview = data.head()
except Exception as e:
    preview, e = None, str(e)
    
preview, e if preview is None else ""