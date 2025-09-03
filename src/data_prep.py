import pandas as pd

def preprocess(input_path: str, output_path: str):
    # Load dataset
    df = pd.read_csv(input_path)

    # Basic cleaning: remove duplicates, reset index
    df = df.drop_duplicates().reset_index(drop=True)

    # Save processed version
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocess("data/Nvidia_stock.csv", "data/Nvidia_stock_processed.csv")
