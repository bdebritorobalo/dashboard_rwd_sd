import pandas as pd 
import numpy as np 

def loading_data(src_path, tgt_path): 
    '''This will load the data with the correct column names as extracted from the HDP.
    :param: src_path: location of the '.rpt' file used as input 
    :param: tgt_path: location of the '.csv' file with the pre-processed data
    '''

    df_raw = pd.read_table(src_path, )