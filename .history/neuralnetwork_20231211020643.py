import pandas as pd

file_path = './data.xlsx'

try:
    data = pd.read_excel(file_path)
    selected_columns = data.columns[:3]
    data_selected = data[selected_columns]
    print(data_selected)
except Exception as e:
    print(e)