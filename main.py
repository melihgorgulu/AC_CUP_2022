# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:23:22 2022

@author: Melih
"""
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from currency_converter import CurrencyConverter
from datetime import date
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing


class Data():
    def __init__(self, data_path=None, df=None, name=None):
        self.data_path = data_path
        try:
            if data_path is None and df is None:
                self.df = -1
                self.columns = -1
                self.col_dtypes = -1
                self.index = -1
                
            elif df is not None and data_path is None:
                self.df = df
                self.columns = self.df.columns
                self.index = self.df.index
                self.col_dtypes = self.df.dtypes
                self.name = name
                
            else:
                self.df = pd.read_csv(data_path)
                self.columns = self.df.columns
                self.index = self.df.index
                self.col_dtypes = self.df.dtypes
                self.name = name
        except Exception as e:
            print('Error while reading the data:')
            print(e)

            
    def __str__(self):
        return 'Data object for : ' + self.data_path
    
    def __len__(self):
        return len(self.df)
    
    def update(self):
            self.columns = self.df.columns
            self.col_dtypes = self.df.dtypes
            self.index = self.df.index
    
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
        self.update()
        
        
    def merge_df(self, right, on=None):
        self.df = pd.merge(left=self.df, right=right, on=on)
        # update
        self.update()
        
    def concat_df(self, df):
        self.df = pd.concat([self.df, df])
        self.update()
        
    def crop_col_data(self, col_name, min_val=None, max_val=None):
        if self.df[col_name].dtype != float or merged_data.df['OFFER_PRICE'].dtype != int:
            print(col_name + 'is not number. Cant crop using min and max values.')
        else:
            if min_val:
                self.df = self.df[self.df[col_name]>min_val]
            if max_val:
                self.df = self.df[self.df[col_name]<max_val]
      
        
        
    def is_col_contain_na(self, col_name=None):
        print('Total na value for col {0}: {1}'.format(col_name, 
                                                       self.df[col_name].isna().sum()))
    
    def get_col_unique_values(self, col_name=None):
        print('Unique values for col {}:'.format(col_name))
        print(self.df[col_name].unique())
        return self.df[col_name].unique()
    
    def get_total_duplicate_of_col(self, col_name=None):
        print('Total duplicate values for col {}:'.format(col_name))
        print(sum(self.df[col_name].duplicated()))
        
    def drop(self, index=None, columns=None):
        self.df.drop(columns=columns,index=index, inplace=True,errors='ignore')
        self._reset_index()
        self.update()
        
    def _reset_index(self):
        self.df.reset_index(drop=True, inplace=True)
        
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
            self.update()
        except Exception as e:
            print('Error while converting data type of ' + str(col_name)+':',e)
            
    def plot_col(self, col_name):
        plt.scatter(np.arange(len(self.df[col_name])), self.df[col_name])
        plt.title('{} for df:{}'.format(col_name,self.name))
        plt.show()
        
    def col_statistics(self, col_name, alpha=0.05):
        x = self.df[col_name]
        print('"'*10)
        print('Statistics for "{}": '.format(col_name))
        print('Mean: ' + str(x.mean()))
        print('Median: ' + str(x.median()))
        print('Std: ' + str(x.std()))
        # shapiro test if data coming from Gaus. Dist.
        # H0: sample x1, ..., xn came from a normally distributed population
        print('Shapiro Test Results:')
        p_val = stats.shapiro(x).pvalue
        print('Shapiro test p_value: {}'.format(p_val))
        if p_val>alpha:
            print('Sample is coming from normal dist.')
        else:
            print('Sample is NOT coming from normal dist.')
        print('"'*10)
        
        

        
class Customer_Data(Data):
    def remove_quotes_curr_revenue(self):
        self.df['REV_CURRENT_YEAR'] = self.df['REV_CURRENT_YEAR'].apply(lambda x: x.replace('"',''))
    
    def convert_country_codes(self):
        self.df['COUNTRY'] = self.df['COUNTRY'].apply(lambda x: 'FR' if x =='France' else 'CH')
    
    def customer_id_update_with_country(self):
        self.df['CUSTOMER'] = self.df['CUSTOMER'].astype(str) + self.df['COUNTRY']
        self.update()
        

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
                x = 'won'
            return x
        self.df['OFFER_STATUS'] = self.df['OFFER_STATUS'].apply(lambda x: fix(x) if x is not np.nan else x)
    def map_isic_data(self):

        ''' map ISIC values to the categories. Reference web-site:https://siccode.com/page/what-is-an-isic-code '''
        def map_isic(x):
            if  x == 0 or x==np.nan:
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
        self.update()
        



def extract_profit(df):
    total_cost = df['MATERIAL_COST']+df['SERVICE_COST']
    return ((df['OFFER_PRICE']-total_cost)*100)/total_cost

def extract_growth(df):
    return (df['REV_CURRENT_YEAR']-df['REV_CURRENT_YEAR.2'])/(np.absolute(df['REV_CURRENT_YEAR.2']+1))


def extract_year_from_mo(x:str)->int:
    x = x.split(' ')[0] # remove hour info
    x = list(map(lambda x: int(x),x.split('.'))) # year will be max. number in date.
    return np.max(x)

if __name__ == '__main__':
    
    ''' This project diveded into 5 section based on CRISP-DM process model '''
    
    ''' 1) _____________________DATA UNDERSTANDING_____________________ '''

    
    '''______CUSTOMER DATA______'''
    # create customer data object
    customer_data = Customer_Data(data_path=r'data/training_data/customers.csv', name='customer_data')
    #print(customer_data.col_dtypes)
    customer_data.remove_quotes_curr_revenue()
    customer_data.convert_col_datatype('REV_CURRENT_YEAR', float)
    #customer_data.data_summary()
    
    '''______TRANSACTION DATA______'''
    trans_data = Transaction_Data(data_path=r'data/training_data/transactions.csv', name='trans_data')
    #trans_data.data_summary()
    # seperate the test data from transaction data.
    #test_data = Data(df=trans_data.df[trans_data.df['TEST_SET_ID'].isnull()==False], name='test_data')
    # save test data.
    #test_data.save_csv('data/test_data.csv', index=False)
    # delete test data from transaction data
    #trans_data.df = trans_data.df[trans_data.df['TEST_SET_ID'].isnull()]
    # drop test_set_id column
    trans_data.drop(columns=['TEST_SET_ID'])
    #trans_data.data_summary()
    trans_data.fix_customer_column()
    trans_data.convert_col_datatype("CUSTOMER", dtype=int)
    # Fix offer status column
    trans_data.fix_offer_status()
    trans_data.merge_df(trans_data.df.groupby('MO_ID')['SO_ID'].count().to_frame().rename(columns={'SO_ID':'N_SUB_OFFER'}), 
                        on='MO_ID')
    # map ISIC data to the categories.
    trans_data.map_isic_data()
    
    '''______GEO DATA______'''
    geo_data = Geo_Data(data_path=r"data/training_data/geo.csv", name='geo_data')
    geo_data.drop_na_row(col_name='SALES_BRANCH')
    #geo_data.data_summary()
    
    ''' Merge transaction and customer data. '''
    # to merge transaction and customer data in a right way, we need to location information
    # for transaction data. We can fetch this via using geo csv.
    trans_data.merge_df(right=geo_data.df[['COUNTRY', 'SALES_LOCATION']], on='SALES_LOCATION')
    # change customer country names to country codes
    customer_data.convert_country_codes()
    # update customer column via merging customer_id and country info for both customer and transaction data.
    customer_data.customer_id_update_with_country()
    # seperate unknown customer_id rows from transaction data
    unk_cust_trans = trans_data.df[trans_data.df['CUSTOMER']==-1]
    # delete unknown customer_id rows from transaction data before updating customer_id
    trans_data.delete_specific_row_by_colval('CUSTOMER',value=-1)

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

    
    merged_data = Data(data_path='data/training_data/merged_data.csv', name='merged_data')
    # Check for na values
    #print(merged_data.na_value_check())
    # fill na values of Rev_current_year with mean grouping by currency
    # before filling it, let's exemine statistics about this column.
    # to get meaningful results, group by currency type first.
    EURO = Data(df=merged_data.df[merged_data.df['CURRENCY']=='Euro'], name='Euro')
    YUAN = Data(df=merged_data.df[merged_data.df['CURRENCY']=='Chinese Yuan'], name='Yuan')
    POUND = Data(df=merged_data.df[merged_data.df['CURRENCY']=='Pound Sterling'], name='Pound')
    DOLLAR = Data(df=merged_data.df[merged_data.df['CURRENCY']=='US Dollar'], name='Usd')
    # Check for statistics
    
    #EURO.col_statistics(col_name='REV_CURRENT_YEAR', alpha=0.05)
    #YUAN.col_statistics(col_name='REV_CURRENT_YEAR', alpha=0.05)
    #POUND.col_statistics(col_name='REV_CURRENT_YEAR', alpha=0.05)
    #DOLLAR.col_statistics(col_name='REV_CURRENT_YEAR', alpha=0.05)
    # For Current Year, for all currency there is big difference between mean and median,
    # and also samples are not coming from normal dist. --> there will be outlier in these data
    
    # detect outliers in "REV_CURRENT_YEAR" via plotting scatter plot
    #EURO.plot_col('REV_CURRENT_YEAR') # crop after 3e6
    #YUAN.plot_col('REV_CURRENT_YEAR') # no need to crop 
    #POUND.plot_col('REV_CURRENT_YEAR') # no need to crop
    #DOLLAR.plot_col('REV_CURRENT_YEAR') # crop after 3e6
    # remove outlier values from "REV_CURRENT_YEAR"
    euro_del_index = EURO.index[EURO.df["REV_CURRENT_YEAR"] >=3*(10**6)]
    dollar_del_index = DOLLAR.index[DOLLAR.df["REV_CURRENT_YEAR"] >=3*(10**6)]
    merged_data.drop(index=euro_del_index)
    merged_data.drop(index=dollar_del_index)
    
    # detect outliers in "REV_CURRENT_YEAR.2" via plotting scatter plot
    #EURO.plot_col('REV_CURRENT_YEAR.1') # crop after 3e6
    #YUAN.plot_col('REV_CURRENT_YEAR.1') # no need to crop 
    #POUND.plot_col('REV_CURRENT_YEAR.1') # no need to crop
    #DOLLAR.plot_col('REV_CURRENT_YEAR.1') # crop after 3e6
    # remove outlier values from "REV_CURRENT_YEAR.1"
    euro_del_index = EURO.index[EURO.df["REV_CURRENT_YEAR.1"] >=3*(10**6)]
    dollar_del_index = DOLLAR.index[DOLLAR.df["REV_CURRENT_YEAR.1"] >=3*(10**6)]
    merged_data.drop(index=euro_del_index)
    merged_data.drop(index=dollar_del_index)
    
    # detect outliers in "REV_CURRENT_YEAR.2" via plotting scatter plot
    #EURO.plot_col('REV_CURRENT_YEAR.2') # no need to crop
    #YUAN.plot_col('REV_CURRENT_YEAR.2') # no need to crop 
    #POUND.plot_col('REV_CURRENT_YEAR.2') # no need to crop
    #DOLLAR.plot_col('REV_CURRENT_YEAR.2') # no need to crop
 
    
    # before filling missing values for Rev_current_year, change currencies to the euro
    # extract year information
    # first make every year format same
    merged_data.df['CREATION_YEAR'] = merged_data.df['CREATION_YEAR'].apply(lambda x: x.replace('-','.') if x is not np.nan else x)
    merged_data.df['CREATION_YEAR'] = merged_data.df['CREATION_YEAR'].apply(lambda x: x.replace('/','.') if x is not np.nan else x)
    merged_data.df['CREATION_YEAR'] = merged_data.df['CREATION_YEAR'].apply(lambda x: int(x.split('.')[-1]) if x is not np.nan else x)
    
    # Currincies: ['Chinese Yuan', 'Pound Sterling', 'Euro', 'US Dollar']

    c = CurrencyConverter(fallback_on_wrong_date=True, fallback_on_missing_rate=True)
    currency_mapper = {'Chinese Yuan':'CNY', 'Pound Sterling':'GBP', 'US Dollar':'USD'}
    for idx, (year, currency, rev, rev_1, rev_2) in enumerate(merged_data.df[['CREATION_YEAR','CURRENCY',
                                                          'REV_CURRENT_YEAR','REV_CURRENT_YEAR.1', 
                                                          'REV_CURRENT_YEAR.2']].to_numpy()):
        if currency=='Euro' or currency is np.nan or year is np.nan:
            continue

        currency = currency_mapper[currency]
        converted_rev = c.convert(rev, currency, date=date(int(year), 1, 1))
        converted_rev1 = c.convert(rev_1, currency, date=date(int(year), 1, 1))
        converted_rev2 = c.convert(rev_2, currency, date=date(int(year), 1, 1))

        
        merged_data.df.at[idx, 'REV_CURRENT_YEAR'] = converted_rev
        merged_data.df.at[idx, 'REV_CURRENT_YEAR.1'] = converted_rev1
        merged_data.df.at[idx, 'REV_CURRENT_YEAR.2'] = converted_rev2
        
    
  
    # when we check na values, it seems like dropping na values would be better
    # drop na values
    merged_data.drop_na_row('REV_CURRENT_YEAR')
    
    # drop uncessassry columns
    drop_cols = ['MO_ID', 'SO_ID','SALES_LOCATION', 'CUSTOMER', 'END_CUSTOMER','REV_CURRENT_YEAR.1','SO_CREATED_DATE']
    merged_data.drop(columns=drop_cols)
    
    ''' 3) _____________________FEATURE EXTRACTION/SELECTION_____________________ '''
    # feature 1: Customer rev. Growth Rate.
    # Growth rate(EndVal-BeginVal)/BeginVal
    merged_data.df['CUSTOMER_GROWTH'] = extract_growth(merged_data.df)
    # feature 2: profit
    merged_data.df['PROFIT'] = extract_profit(merged_data.df)
    # feature 3: how long customer do know the company
    merged_data.df['MO_CREATED_DATE'] = merged_data.df['MO_CREATED_DATE'].apply(lambda x: x.replace('-','.') if x is not np.nan else x)
    merged_data.df['MO_CREATED_DATE'] = merged_data.df['MO_CREATED_DATE'].apply(lambda x: x.replace('/','.') if x is not np.nan else x)
    merged_data.df['MO_CREATED_DATE'] = merged_data.df['MO_CREATED_DATE'].apply(lambda x: extract_year_from_mo(x))
    
    merged_data.df['CUSTOMER_HOW_LONG'] = merged_data.df['MO_CREATED_DATE'] - merged_data.df['CREATION_YEAR']
    
    # delete negative cols.
    merged_data.df = merged_data.df[merged_data.df['CUSTOMER_HOW_LONG']>=0]
    # delete cols
    merged_data.drop(columns=['CREATION_YEAR', 'MO_CREATED_DATE'])
    
    # feature 4: Which one is costly Meterial or service
    merged_data.df['SERVICE_COSTLY'] = list(map(lambda x: 1 if x>0 else 0, list(merged_data.df['SERVICE_COST'] - merged_data.df['MATERIAL_COST'])))
    
    
    # check for correlation matrix
    corr = merged_data.df.corr()
    # multicollinearity between SERVICE_LIST_PRICE and MATERIAL_COST
    # multicollinearity between REV_CUR_YEAR and REV_CUR_YEAR.2
    # multicollinearity between REV_CUR_YEAR and REV_CUR_YEAR.2
    # multicollinearity between OFFER PRICE-SERVICE_LIST_PRICE and MATERIAL_COST
    # drop these columns
    
    merged_data.drop(columns=['SERVICE_LIST_PRICE', 'MATERIAL_COST', 'SERVICE_COST'])
    
    # check corr again
    corr = merged_data.df.corr() # looks nice
    
    # apply one hot encoder
    cols = ['PRICE_LIST', 'ISIC', 'TECH', 'OFFER_TYPE','BUSINESS_TYPE','OWNERSHIP', 'CURRENCY']

    merged_data.df = pd.get_dummies(merged_data.df, columns=cols)
    
    # binerize label
    
    lb = preprocessing.LabelBinarizer()
    merged_data.df['COUNTRY'] = lb.fit_transform(merged_data.df['COUNTRY'])

    
    # For numeric input, use ANOVA
    # categorical feature selection
    # use Chi-Squared Statistic for categoric input
    
    #fs = SelectKBest(score_func=chi2, k='all')
    #y = merged_data.df[['OFFER_STATUS']]
    #X = merged_data.df.drop(['OFFER_STATUS'],axis=1)
    #fs.fit(X, y)
    
    # save merged_data
    #merged_data.save_csv(path='data/training_data/final_data.csv')
    
    
    
    


 
    
    
    
    
    
    
    
    
    



    
  