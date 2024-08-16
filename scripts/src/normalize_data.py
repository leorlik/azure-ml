from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from pathlib import Path


def argument_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data', type=str, help='Input CSV file path')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of data to be used as test set')
    parser.add_argument('--output_train', type=str, help='Path to save the training dataset')
    parser.add_argument('--output_test', type=str, help='Path to save the testing dataset')

    args = parser.parse_args()

    return args.input_data, args.test_size , args.output_train, args.output_test
    

input_data, test_size, output_train, output_test = argument_parser()

# Load data
data = pd.read_csv(input_data)

# Perform train/test split
train_data, test_data = train_test_split(data, test_size=test_size, random_state=98)

# Save datasets
train_data.to_csv(output_train, index=False)
test_data.to_csv(output_test, index=False)