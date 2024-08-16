import pandas as pd
import argparse
from pathlib import Path


def argument_parser():

    parser = argparse.ArgumentParser(description='Optional app description')
    
    parser.add_argument('--columns_to_drop', type=list,
                    help='list to arguments to one_hot_encode')
    
    parser.add_argument("--input_data", type = str,
                        help = 'the data to encode')
    
    parser.add_argument("--output_data_path", type = str,
                        help = 'data to save the argument ')
    
    args = parser.parse_args()
    
    return args.input_data, args.columns_to_encode, args.output_data_path

input_data, columns_to_drop, output_data_path = argument_parser()

df = pd.read_csv(input_data)

df.drop(columns = columns_to_drop.split(" "), inplace = True)

output_df = df.to_csv(
    (Path(output_data_path) / "droped-data.csv"), 
    index = False
)

output_data = output_df