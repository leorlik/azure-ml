import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='Input CSV file path')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to be used as test set')
    parser.add_argument('--output_train', type=str, help='Path to save the training dataset')
    parser.add_argument('--output_test', type=str, help='Path to save the testing dataset')

    args = parser.parse_args()

    # Load data
    data = pd.read_csv(args.input_data)

    # Perform train/test split
    train_data, test_data = train_test_split(data, test_size=args.test_size)

    # Save datasets
    train_data.to_csv(args.output_train, index=False)
    test_data.to_csv(args.output_test, index=False)

if __name__ == '__main__':
    main()