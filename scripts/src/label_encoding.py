from sklearn.preprocessing import LabelEncoder
import pandas as pd
import argparse
from pathlib import Path


def argument_parser():

    parser = argparse.ArgumentParser(description='Optional app description')
    
    parser.add_argument('--columns_to_encode', type=list,
                    help='list to arguments to one_hot_encode')
    
    parser.add_argument("--input_data", type = str,
                        help = 'the data to encode')
    
    parser.add_argument("--output_data_path", type = str,
                        help = 'data to save the argument ')
    
    args = parser.parse_args()
    
    return args.input_data, args.columns_to_encode, args.output_data_path

input_data, columns_to_encode, output_data = argument_parser()

df = pd.read_csv(input_data)

lbl = LabelEncoder()

for c in columns_to_encode():

    df[c] = lbl.fit_transform(df[c])

# save the data as a csv
output_df = df.to_csv(
    (Path(output_data) / "encoded-data.csv"), 
    index = False
)

output_data = output_df
    
