"""This module contains all functions created for the pre-processing,
including loading and saving the data.
"""

import uuid
import numpy as np
import pandas as pd

class PREPROC:
    '''All functions concernig preprocessing, more information will follow'''
    def __init__(self):
        self.header_names = ['surgery_id', 'subject_id', 'time_OR', 'postop_diagnosis_code',                #4
                        'postop_diagnosis_text', 'procedure_code', 'procedure_text', 'main_procedure',      #8
                        'procedure_duration', 'procedure_duration_fix','age_procedure', 'status_sternum',   #12
                        'ECC_duration', 'AOX_duration','DHCA_duration', 'ACP_duration']                     #16
        self.column_numbers=[0,1,5,8,9,10,11,12,15,16,17,18,19,20,21,22]
        self.removable_procedures=['989999', '339121', '339130B', '332486A', '339132', '332429',
                                   '335512C', '332281A','332280A','333180B','332215D']
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


    def remove_procedures(self, extra_procedure=None):
        '''
        This function will remove a preset list of procedures(for example 'procedure delayed')
        The standard set of procedures to exclude is set in the class.
        :param: extra_procedure: Add an extra list or singular procedure(s) to exclude.
        :returns: data: Returns an updated/filtered version of the dataframe. Stored in class.
        '''
        df          = self.data
        procedures  = self.removable_procedures

        # 1. Plain procedures to remove
        if extra_procedure:
            if isinstance(extra_procedure, list):
                for proc in extra_procedure:
                    procedures.append(proc)
            elif isinstance(extra_procedure, str):      # | isinstance(extra_procedure, int): not needed?
                procedures.append(extra_procedure)

        for proc in procedures:
            condition0 = df['procedure_code'] == proc
            df = df[~condition0]

        # 2. Multiple conditions (manually added)
        # Specific condition where donor heart was explanted, in external hospital.
        condition1 = df['time_OR'] == 0
        condition2 = df['procedure_code'] == '332553A'  # explanation donor heart (external hosp.)
        condition = condition1 & condition2
        df = df[~condition]

        # 3. Remove 'rare' procedures
        value_counts = df['procedure_code'].value_counts()
        df_new = df[df['procedure_code'].isin(value_counts[value_counts >= 5].index)]

        # 4. Remove 'rare' post-op diagnoses
        value_counts = df_new['postop_diagnosis_code'].value_counts()
        df_filtered = df_new[df_new['postop_diagnosis_code'].isin(value_counts[value_counts >= 5].index)]

        self.data = df_filtered

    def fix_procedure_duration(self):
        '''
        Due to poor data insertion in the source data, sometimes procedure times are missing.
        Goal is to remove the impossible date-times. Further solutions should be explored in raw data.
        :returns: updated data: three columns are used for creating most accurate procedure_duration,
                                unused columns are droped
        '''
        df = self.data
        condition1 = df['time_OR']> 600
        condition2 = df['procedure_duration'] <= 0
        condition3 = df['procedure_duration_fix'] > 500

        condition = condition1 & condition2 & condition3

        df = df[~condition]

        df['procedure_duration'] = df[['time_OR',
                                       'procedure_duration',
                                       'procedure_duration_fix']].replace(0, np.nan).abs().min(axis=1)

        self.data = df #df.drop(columns=['procedure_duration_fix', 'time_OR']).reset_index()



if __name__ == '__main__':
    # will be moved towards main.py
    data = PREPROC()
    # data.read_data('/Volumes/Brian/PHEMS/veilai/dashboard_rwd_sd/data/raw/20240529_all_procedures.rpt')
    data.read_data('/Volumes/Brian/PHEMS/veilai/dashboard_rwd_sd/data/raw/20240603_CTC_procedures.txt')
    data.pseudonymize('/Volumes/Brian/PHEMS/veilai/dashboard_rwd_sd/data/config/pid_keyfile.pkl', 'subject_id')
    data.pseudonymize('/Volumes/Brian/PHEMS/veilai/dashboard_rwd_sd/data/config/surgery_keyfile.pkl', 'surgery_id')

    data.remove_procedures()
    data.fix_procedure_duration()