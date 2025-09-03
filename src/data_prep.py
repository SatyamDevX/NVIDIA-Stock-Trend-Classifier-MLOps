import pandas as pd

def preprocess(input_path: str, output_path: str):
    # Load dataset
    df = pd.read_csv(input_path)

    # Basic cleaning: remove duplicates, reset index
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Drop Date column (not useful directly for ML)
    if "Date" in df.columns:
        df = df.drop(columns=["Date"])


    # Create target: 1 if next day's close is higher than today's, else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop last row because its Target will be NaN (due to shift)
    df = df.dropna()

    # Save processed version
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}, shape={df.shape}")

if __name__ == "__main__":
    preprocess("data/Nvidia_stock.csv", "data/Nvidia_stock_processed.csv")
