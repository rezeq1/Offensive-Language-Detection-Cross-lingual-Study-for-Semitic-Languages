import pandas as pd

def readFileFunction(path):
    # result = None
    filepath = path.split('.')
    end=filepath[-1]
    if end == 'csv':
        data = pd.read_csv(path)
    if end == 'xlsx':
        data = pd.read_excel(path)
    
    return data
    


