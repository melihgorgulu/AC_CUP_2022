# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:23:22 2022

@author: Melih
"""
# Importing Libraries
import pandas as pd
import numpy as np



class Data():
    def __init__(self, data_path=None, df=None, name=None):
        self.data_path = data_path
        try:
            if data_path is None and df is None:
                self.df = -1
                self.columns = -1
                self.col_dtypes = -1
                
            elif df is not None and data_path is None:
                self.df = df
                self.columns = self.df.columns
                self.col_dtypes = self.df.dtypes
                self.name = name
                
            else:
                self.df = pd.read_csv(data_path)
                self.columns = self.df.columns
                self.col_dtypes = self.df.dtypes
                self.name = name
        except Exception as e:
            print('Error while reading the data:')
            print(e)

            
    def __str__(self):
        return 'Data object for : ' + self.data_path
    
    def save_csv(self, path, index=False):
        self.df.to_csv(path, index=index)
    
    def describe(self, include=None, exclude=None):
        return self.df.describe(include=include, exclude=exclude)
    
    def na_value_check(self):
        print('Number of N.a value in the dataframe "{0}":'.format(self.name))
        print(self.df.isna().sum())
        
    
    def drop_na_row(self, col_name):
        self.df = self.df[self.df[col_name].notna()]
        
    def drop(self, index=None, columns=None):
        self.df.drop(columns=columns, inplace=True)
        self.columns = self.df.columns
        self.col_dtypes = self.df.dtypes
        
        
    def data_summary(self, include=None, exclude=None):
        print('Basic data statistics of "{0}":'.format(self.name))
        print('='*15)
        print(self.df.describe(include, exclude))
        print('='*15)
        self.na_value_check()
        print('='*15)
    
    def rename_columns(self, mapper:dict):
        try:
            self.df = self.df.rename(columns=mapper)
            self.columns = self.df.columns
        except Exception as e:
            print('Error while renaming columns:', e)
    

    def convert_col_datatype(self, col_name, dtype):
        try:
            self.df = self.df.astype({col_name: dtype})
            # update columns dtype
            self.col_dtypes = self.df.dtypes
        except Exception as e:
            print('Error while converting data type of ' + str(col_name)+':',e)

        
class Customer_Data(Data):
    def remove_quotes_curr_revenue(self):
        self.df['REV_CURRENT_YEAR'] = self.df['REV_CURRENT_YEAR'].apply(lambda x: x.replace('"',''))
        

class Transaction_Data(Data):
    def fix_customer_column(self):
        ''' Remove quotes and replace NA/#NV with -1 '''
        self.df['CUSTOMER'] = self.df['CUSTOMER'].apply(lambda x: x.replace('"',''))
        self.df['CUSTOMER'] = self.df['CUSTOMER'].apply(lambda x: x.replace('#NV','-1'))
        self.df['CUSTOMER'] = self.df['CUSTOMER'].apply(lambda x: x.replace('NA','-1'))

if __name__ == '__main__':
    
    # This project diveded into 5 section based on CRISP-DM process model
    
    # 1) Data Understanding

    
    '''______CUSTOMER DATA______'''
    # create customer data object
    customer_data = Customer_Data(data_path=r'data/training_data/customers.csv', name='customer_data')
    #print(customer_data.col_dtypes)
    customer_data.rename_columns({'CUSTOMER':'CUSTOMER_ID'})
    customer_data.remove_quotes_curr_revenue()
    customer_data.convert_col_datatype('REV_CURRENT_YEAR', float)
    customer_data.data_summary()
    
    '''______TRANSACTION DATA______'''
    trans_data = Transaction_Data(data_path=r'data/training_data/transactions.csv', name='trans_data')
    trans_data.data_summary()
    # seperate the test data from transaction data.
    test_data = Data(df=trans_data.df[trans_data.df['TEST_SET_ID'].isnull()==False], name='test_data')
    # save test data.
    test_data.save_csv('data/test_data.csv', index=False)
    # delete test data from transaction data
    trans_data.df = trans_data.df[trans_data.df['TEST_SET_ID'].isnull()]
    # drop test_set_id column
    trans_data.drop(columns=['TEST_SET_ID'])
    trans_data.data_summary()
    trans_data.fix_customer_column()
    trans_data.convert_col_datatype("CUSTOMER", dtype=int)
    
    





    
  