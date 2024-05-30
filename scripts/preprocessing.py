"""This module contains all functions created for the pre-processing,
including loading and saving the data.
"""

import uuid
import pandas as pd

class PREPROC:
    '''All functions concernig preprocessing, more information will follow'''
    def __init__(self):
        self.header_names = ['surgery_id', 'subject_id', 'time_OR', 'postop_diagnosis_code',
                        'postop_diagnosis_text', 'procedure_code', 'procedure_text', 'main_procedure',
                        'procedure_duration', 'procedure_duration_fix','age_procedure', 'status_sternum',
                        'ECC_duration', 'AOX_duration','DHCA_duration', 'ACP_duration']
        self.column_numbers=[0,1,5,8,9,10,11,12,15,16,17,18,19,20,21,22]
        self.data = None

    def read_data(self, src_path):
        '''
        This will load the data with the correct column names as extracted from the HDP.
        :param: src_path: location of the '.rpt' file used as input
        :returns: data  : Within class, dataframe is stored.
        '''
        self.data = pd.read_table(src_path, header=0, usecols=self.column_numbers,
                                  names=self.header_names)

    def pseudonymize(self, tgt_path, column_name):
        '''Function to pseudonymize data.
        :param: tgt_path: File path (including file name) to store the keyfile between original and pseudonymized ID
        :param: column_name: Name of the column that must be pseudonymized.
        :returns: keyfile: Pickle-file that store the original and pseudonymized values for the IDs per column.
        :returns: data   : Data in class is overwritten with pseudonymized version of the data/column
        '''
        temp_data = self.data

        raw = temp_data[column_name].unique()
        pseudo = [uuid.uuid4().hex for _ in raw]

        if len(set(pseudo)) == len(set(raw)):              # check for number of unique values, should be equal
            print('No copies created while generating pseudonymized codes, all good to continue')
                        # otherwise same studyID generated (very unlikely)

            temp_data[column_name].replace(dict(zip(raw, pseudo)), inplace=True)        # replace patientID with studyID

            pd.DataFrame({f'{column_name}_original':raw,
                          f'{column_name}_pseudo': pseudo}).to_pickle(tgt_path)         #('pID_stID_keys.pkl')
        else:
            print('WARNING, there are duplicates in the studyId list')

        self.data= temp_data
        return temp_data


    def procedure_duration(self):
        '''Due to poor data insertion in the source data, sometimes procedure times are missing.
        Goal is to remove the impossible date-times.
        :param: still figuring it out: 
        '''
        



if __name__ == '__main__':
    # will be moved towards main.py
    data = PREPROC()
    data.read_data('/Volumes/Brian/PHEMS/veilai/dashboard_rwd_sd/data/raw/20240529_all_procedures.rpt')
    data.pseudonymize('/Volumes/Brian/PHEMS/veilai/dashboard_rwd_sd/data/config/pid_keyfile.pkl', 'subject_id')
    data.pseudonymize('/Volumes/Brian/PHEMS/veilai/dashboard_rwd_sd/data/config/surgery_keyfile.pkl', 'surgery_id')
