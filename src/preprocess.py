import pandas as pd
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove everything except letters
    return text

def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    df['clean_text'] = df['Review Texts'].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_and_preprocess("data/dataframe_with_category_modified.csv")
    print(df.head())