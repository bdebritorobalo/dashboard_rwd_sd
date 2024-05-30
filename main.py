'''Main python code for data pre-processing'''

from scripts.preprocessing import PREPROC

def pre_processing_pipeline(src_data):
    '''another string, pylinting/docstrings sucks sometimes'''
    data = PREPROC()
    data.read_data(src_data)
    return data

if __name__ == '__main__':
    df = pre_processing('data/20240529_all_procedures.rpt')
    print(df.head())
