# Author : wfarhani@lumenai.fr
# Date : 2019-02-13 17:09
# Project: ididentifier
 
# imports
import os
import pandas as pd
import numpy as np
from itertools import zip_longest, tee
import scipy.stats as stats
 
 
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
 
 
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
 
 
def genereate_dfs(file_path, n=5):
    """
    Generator of dataframes from the input file
    :param file_path: the file path/name
    :param n: number of splits
    :return generator of dataframes
    """
    # load file as pandas.DataFrame
    file_df = pd.read_csv(file_path, delimiter=',', header=0)
 
    # split file_df into n subset
    separators = [l[0] for l in list(grouper(file_df.transpose(), int(np.ceil(len(file_df)/n))))]
    list_of_subdf = [file_df[s:e] if e is not None else file_df[s:] for s, e in pairwise(separators)]
 
    # iterate through dataframes
    for subdf in list_of_subdf:
        # iterate trough subdf columns
        for col in subdf.columns:
            if len(subdf[col].unique()) == 1:  # if the column is constant
                # than drop it
                subdf.drop(col, axis=1, inplace=True)
            else:
                if not np.issubdtype(
                        subdf[col].dtype, np.number
                ) and subdf[col].dtype != bool:  # if the column is not numerical
 
                    if len(subdf[col].unique()) < len(subdf)*0.2:  # if the column has low granularity => categorical variable
                        # convert categorical data to numeric
                        to_rep = dict(zip(subdf[col].unique(), range(len(subdf[col].unique()))))
                        subdf.replace({col: to_rep}, inplace=True)
 
                    else:  # if the column has low granularity => not categorical
                        # drop the column
                        subdf.drop(col, axis=1, inplace=True)
                if col in subdf.columns:  # if the column still there
                    # drop nans
                    subdf[col].fillna(subdf[col].mean(), inplace=True)
                    # convert column to float
                    subdf[col] = subdf[col].astype(float)
        yield subdf
 
 
def compute_columns(df, filename):
    """
    Compute columns of each attribute
    :param df: the input pandas.DataFrames
    :param filename: source file name
    :return: Pandas series
    """
    rows = list(
                df.apply(lambda c: {
                    'position': list(df.columns).index(c.name),
                    'name': c.name,
                    'filename': filename,
                    'unique_ratio': len(c.unique())/len(df),
                    'entropy': stats.entropy(c),
                    'max': c.max(),
                    'min': c.min(),
                    'std': c.std(),
                    'mean': c.mean(),
                    'p25': c.quantile(.25),
                    'p50': c.quantile(.5),
                    'p75': c.quantile(.75),
                    'pearson_corr': df.corr()[c.name].sum(),
                    'spearman_corr': df.corr(method='spearman')[c.name].sum(),
                    'is_id': 0
                })
            )
    return pd.DataFrame(rows)
 
 
if __name__ == '__main__':
    # YOUR DATASETS DIRECTORY HERE
    directory = os.fsencode('datasets')
    
    # Create empty pandas dataframe
    final_dataset = pd.DataFrame(columns=['name', 'unique_ratio', 'entropy', 'max', 'min', 'std', 'mean', 'p25'])
    for file in os.listdir(directory):  # iterate through datasets
        # open filename
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            # iterate through sub-dataframes
            for df in genereate_dfs('datasets/{0}'.format(filename)):
                file_columns = compute_columns(df, filename)
                final_dataset = pd.concat([final_dataset, file_columns], ignore_index=True)
        else:
            continue
    # store the final dataset as .csv file
    final_dataset.to_csv('datasets/final_dataset.csv', sep=',', header=True, index=False)
