import pandas as pd
import numpy as np

df = pd.read_csv('data/TRAINING/training_no_bad_lines_val.csv')

df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.5

train = df[msk]
test = df[~msk]

train.to_csv('data/TRAINING/training_no_bad_lines_val.csv')

test.to_csv('data/TRAINING/training_no_bad_lines_test.csv')