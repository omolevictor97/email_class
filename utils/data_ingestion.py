import pandas as pd
from sklearn.model_selection import train_test_split

def data_ingest(file):
    try:
        df = pd.read_csv(file, sep="\t", header=None)
        df.columns = ["Labels", "Messages"]

        #Let's do some simple splitting, so as to have some test dataset for referencing
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        test_df.to_csv('test.csv')
        return df
    except Exception as e:
        return str(e)
