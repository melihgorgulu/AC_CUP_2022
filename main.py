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
    
    def __len__(self):
        return len(self.df)
    
    def _update(self):
            self.columns = self.df.columns
            self.col_dtypes = self.df.dtypes
    
    def save_csv(self, path, index=False):
        self.df.to_csv(path, index=index)
    
    def describe(self, include=None, exclude=None):
        return self.df.describe(include=include, exclude=exclude)
    
    def na_value_check(self):
        print('Number of N.a value in the dataframe "{0}":'.format(self.name))
        print(self.df.isna().sum())
        
    
    def drop_na_row(self, col_name):
        self.df = self.df[self.df[col_name].notna()]
    
    def delete_specific_row_by_colval(self, col_name, value):
        self.df = self.df[self.df[col_name]!=value]
        self._update()
        
    def merge_df(self, right, on=None):
        self.df = pd.merge(left=self.df, right=right, on=on)
        # update
        self._update()
        
    def concat_df(self, df):
        self.df = pd.concat([self.df, df])
        self._update()
        
    def is_col_contain_na(self, col_name=None):
        print('Total na value for col {0}: {1}'.format(col_name, 
                                                       self.df[col_name].isna().sum()))
    
    def get_col_unique_values(self, col_name=None):
        print('Unique values for col {}:'.format(col_name))
        print(self.df[col_name].unique())
    
    def get_total_duplicate_of_col(self, col_name=None):
        print('Total duplicate values for col {}:'.format(col_name))
        print(sum(self.df[col_name].duplicated()))
        
    def drop(self, index=None, columns=None):
        self.df.drop(columns=columns, inplace=True)
        self._update()
        
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
            self._update()
        except Exception as e:
            print('Error while converting data type of ' + str(col_name)+':',e)

        
class Customer_Data(Data):
    def remove_quotes_curr_revenue(self):
        self.df['REV_CURRENT_YEAR'] = self.df['REV_CURRENT_YEAR'].apply(lambda x: x.replace('"',''))
    
    def convert_country_codes(self):
        self.df['COUNTRY'] = self.df['COUNTRY'].apply(lambda x: 'FR' if x =='France' else 'CH')
    
    def customer_id_update_with_country(self):
        self.df['CUSTOMER'] = self.df['CUSTOMER'].astype(str) + self.df['COUNTRY']
        self._update()
        

class Geo_Data(Data):
    pass

class Transaction_Data(Data):
    
    def fix_customer_column(self):
        ''' Remove quotes and replace NA/#NV with -1 '''
        self.df['CUSTOMER'] = self.df['CUSTOMER'].apply(lambda x: x.replace('"',''))
        self.df['CUSTOMER'] = self.df['CUSTOMER'].apply(lambda x: x.replace('#NV','-1'))
        self.df['CUSTOMER'] = self.df['CUSTOMER'].apply(lambda x: x.replace('NA','-1'))
        
    def fix_offer_status(self):
        ''' fix the offer entries '''
        def fix(x):
            x = x.lower()
            if x == 'lose':
                x = 'lost'
            elif x == 'win':
                x == 'won'
            return x
        self.df['OFFER_STATUS'] = self.df['OFFER_STATUS'].apply(lambda x: fix(x))
    def map_isic_data(self):

        ''' map ISIC values to the categories. Reference web-site:https://siccode.com/page/what-is-an-isic-code '''
        def map_isic(x):
            if  x == 0:
                return 'X'
            else:
                # exctract division
                div = int(str(x)[:2])
                if div >=10 and div <=33:
                    return 'C'
    
                elif div==35:
                    return 'D'
                elif div>=36 and div <=40:
                    return 'E'
                elif div>=41 and div <=43:
                    return 'F'
                elif div>=45 and div <=47:
                    return 'G'
                elif div>=49 and div <=53:
                    return 'H'
                elif div>=55 and div <=56:
                    return 'I'
                elif div>=58 and div <=63:
                    return 'J'
                elif div>=64 and div <=66:
                    return 'K'
                elif div==68:
                    return 'L'
                elif div>=69 and div <=75:
                    return 'M'
                elif div>=77 and div <=82:
                    return 'N'
                elif div==84:
                    return 'O'
                elif div==85:
                    return 'P'
                elif div>=86 and div <=88:
                    return 'Q'
                elif div>=89 and div <=93:
                    return 'R'
                elif div>=94 and div <=99:
                    return 'U'

            return div
            
        # fill na values with 0
        value = {'ISIC':0}
        self.df = self.df.fillna(value=value)
        self.df['ISIC'] = self.df['ISIC'].apply(lambda x: map_isic(x))
        
    def customer_id_update_with_country(self):
        self.df['CUSTOMER'] = self.df['CUSTOMER'].astype(str) + self.df['COUNTRY']
        self._update()
        
        

if __name__ == '__main__':
    
    ''' This project diveded into 5 section based on CRISP-DM process model '''
    
    ''' 1) _____________________DATA UNDERSTANDING_____________________ '''

    
    '''______CUSTOMER DATA______'''
    # create customer data object
    customer_data = Customer_Data(data_path=r'data/training_data/customers.csv', name='customer_data')
    #print(customer_data.col_dtypes)
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
    trans_data.convert_col_datatype("CUSTOMER_ID", dtype=int)
    # Fix offer status column
    trans_data.fix_offer_status()
    trans_data.merge_df(trans_data.df.groupby('MO_ID')['SO_ID'].count().to_frame().rename(columns={'SO_ID':'N_SUB_OFFER'}), 
                        on='MO_ID')
    # map ISIC data to the categories.
    trans_data.map_isic_data()
    
    '''______GEO DATA______'''
    geo_data = Geo_Data(data_path=r"data/training_data/geo.csv", name='geo_data')
    geo_data.drop_na_row(col_name='SALES_BRANCH')
    geo_data.data_summary()
    
    ''' Merge transaction and customer data. '''
    # to merge transaction and customer data in a right way, we need to location information
    # for transaction data. We can fetch this via using geo csv.
    trans_data.merge_df(right=geo_data.df[['COUNTRY', 'SALES_LOCATION']], on='SALES_LOCATION')
    # change customer country names to country codes
    customer_data.convert_country_codes()
    # update customer column via merging customer_id and country info for both customer and transaction data.
    customer_data.customer_id_update_with_country()
    # seperate unknown customer_id rows from transaction data
    unk_cust_trans = trans_data.df[trans_data.df['CUSTOMER']=='-1']
    # delete unknown customer_id rows from transaction data before updating customer_id
    trans_data.delete_specific_row_by_colval('CUSTOMER',value='-1')

    #  update customer id
    trans_data.customer_id_update_with_country()
    
    merged_data = Data(df=pd.merge(left=trans_data.df, right=customer_data.df, on='CUSTOMER',suffixes=('','_y')), name='merged_df')
    merged_data.drop(columns=['COUNTRY_y'])
    # add unknown customer transactions to the merge data frame
    for col in customer_data.columns:
        if col =='COUNTRY' or col=='CUSTOMER':
            continue
        unk_cust_trans[col] = np.nan
        
    merged_data.concat_df(unk_cust_trans)
    
    # save merged data
    merged_data.save_csv('data/training_data/merged_data.csv')
    
    
    
    
    ''' 2) _____________________DATA PREPARATION_____________________ '''
    
    
    
    
    
    
    
    
    



    
  