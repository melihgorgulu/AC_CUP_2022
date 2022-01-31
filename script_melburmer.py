# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:23:22 2022

@author: Melih

GITHUB REPO FOR THIS PROJECT : https://github.com/melburmer/AC_CUP_2022

"""
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from currency_converter import CurrencyConverter
from datetime import date
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
        
        
    def merge_df(self, right, on=None, how='inner'):
        self.df = pd.merge(left=self.df, right=right, on=on, how=how)
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
    customer_data = Customer_Data(data_path=r'customers.csv', name='customer_data')
    #print(customer_data.col_dtypes)
    customer_data.remove_quotes_curr_revenue()
    customer_data.convert_col_datatype('REV_CURRENT_YEAR', float)
    #customer_data.data_summary()
    
    '''______TRANSACTION DATA______'''
    trans_data = Transaction_Data(data_path=r'transactions.csv', name='trans_data')
    #trans_data.data_summary()
    # seperate the test data from transaction data.

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
    geo_data = Geo_Data(data_path=r"geo.csv", name='geo_data')
    geo_data.drop_na_row(col_name='SALES_BRANCH')
    #geo_data.data_summary()
    
    ''' Merge transaction and customer data. '''
    # to merge transaction and customer data in a right way, we need to location information
    # for transaction data. We can fetch this via using geo csv.
    trans_data.merge_df(right=geo_data.df[['COUNTRY', 'SALES_LOCATION']], on='SALES_LOCATION', how='left')
    
    # fill country and sales location
    trans_data.df['COUNTRY'].fillna(method="ffill", inplace=True)
    trans_data.df['SALES_LOCATION'].fillna(method="ffill", inplace=True)
    
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

    
    merged_data = Data(df=pd.merge(left=trans_data.df, right=customer_data.df, on='CUSTOMER',suffixes=('','_y'), how='left'), name='merged_df')
    merged_data.drop(columns=['COUNTRY_y'])


    # add unknown customer transactions to merge the dataframe
    for col in customer_data.columns:
        if col =='COUNTRY' or col=='CUSTOMER':
            continue
        unk_cust_trans[col] = np.nan
        
    merged_data.concat_df(unk_cust_trans)
    
    
    
    # save merged data
    #merged_data.save_csv('data/training_data/merged_data.csv')
    
    
    ''' 2) _____________________DATA PREPARATION_____________________ '''
    
    # first split test data
    test_data = Data(df=merged_data.df[merged_data.df['TEST_SET_ID'].isnull()==False], name='test_data')
    
    #merged_data = Data(data_path='data/training_data/merged_data.csv', name='merged_data')
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

    # Use CurrencyConverter for make it easy.
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
    # delete some cols
    merged_data.drop(columns=['CREATION_YEAR', 'MO_CREATED_DATE'])
    
    # feature 4: Which one is costly- Meterial or service?
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
    
    data_backup_to_fill_na = merged_data.df.copy()
    merged_data.df = pd.get_dummies(merged_data.df, columns=cols)
    
    # binerize label
    lb = preprocessing.LabelBinarizer()
    merged_data.df['COUNTRY'] = lb.fit_transform(merged_data.df['COUNTRY'])
    
    # delete test data from merged data
    merged_data.df = merged_data.df[merged_data.df['TEST_SET_ID'].isnull()]
    # drop test_set_id column
    merged_data.drop(columns=['TEST_SET_ID'])
    # split merged data as depentent and indepentent variable.
    
    y = merged_data.df[['OFFER_STATUS']]
    X = merged_data.df.drop(['OFFER_STATUS'], axis=1)
   
    
    # manually encode depentent variable values. 1: won , 0: loss 
    y['OFFER_STATUS'] = y['OFFER_STATUS'].apply(lambda x: 1 if x=='won' else 0)
    
    # we have 92 feature for X, we can reduce it.
    ''' **Lets do feature selection** '''
    # reference -> https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
    # check correlation matrix again.
    corr = X.corr() # looks nice
    
    ''' For numeric input-categorical output we can use ANOVA.'''
    # Let's first differentiate categorical and numeric data
    numeric_columns = ['OFFER_PRICE', 'COSTS_PRODUCT_A', 'COSTS_PRODUCT_B', 
                       'COSTS_PRODUCT_C', 'COSTS_PRODUCT_D', 'COSTS_PRODUCT_E', 
                       'N_SUB_OFFER','REV_CURRENT_YEAR', 'REV_CURRENT_YEAR.2','CUSTOMER_GROWTH',
                       'PROFIT', 'CUSTOMER_HOW_LONG']
    numeric_data_x = X[numeric_columns]
    categorical_data_x = X.drop(numeric_columns, axis=1)
    
    # Min-max normalization before run ANOVA test with numeric data
    Min_Max_Scaler = preprocessing.MinMaxScaler()
    numeric_data_x = pd.DataFrame(Min_Max_Scaler.fit_transform(numeric_data_x), columns=numeric_columns)
    # apply Anova Test with SelectKBest.
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif
    anova_fs = SelectKBest(score_func=f_classif, k=11)
    anova_fs.fit(numeric_data_x, y)
    selected_cols = anova_fs.get_support(indices=True)
    selected_numeric_data_x = numeric_data_x.iloc[:, selected_cols]
    # find un-selected columns by anova test
    unwanted_cols_anova = list(set(numeric_data_x) - set(selected_numeric_data_x.columns))
    # for now, just keep all numeric data in the X.
    anova_scores = anova_fs.scores_
    # score for PROFIT is too low, we can delete this feature, to do this, set k=11.
    
    ''' For categorical input-categorical output we can use Chi-Squared Statistic'''
    from sklearn.feature_selection import chi2
    chi_fs = SelectKBest(score_func=chi2, k='all')
    chi_fs.fit(categorical_data_x, y)
    # check for scores
    chi_scores = chi_fs.scores_
    # plot scores to determine param k for SelectKBest
    # plot the scores
    #plt.bar([i for i in range(len(chi_scores))], chi_scores)
    # plot threshold
    th = [10 for i in range(len(chi_scores))] # threshold 10 looks nice
    #plt.plot(np.arange(len(chi_scores)),th, color='red')
    #plt.show()
    # delete all features which score is lower than th
    number_of_features_to_delete = np.sum(np.where(chi_scores < th, 1, 0))
    # we can delete like 43 features, so set param k = 80-43=37
    chi_fs = SelectKBest(score_func=chi2, k=37)
    chi_fs.fit(categorical_data_x, y)
    selected_cols = chi_fs.get_support(indices=True)
    selected_categorical_data_x = categorical_data_x.iloc[:, selected_cols]
    # find un-selected columns by chi square test
    unwanted_cols_chi = list(set(categorical_data_x) - set(selected_categorical_data_x.columns))
    
    # Rearrange indepentent variable X with selected columns.
    cols_to_delete = unwanted_cols_anova + unwanted_cols_chi
    cols_to_delete.append('CUSTOMER_HOW_LONG')
    X = X.drop(cols_to_delete, axis=1)
    # also drop 
    
    
    ''' 4) _____________________MODELING_____________________ '''
 
    # Let's do train-validation split first.
    # before doing split, lets check our target variables frequencies
    print(y.value_counts())  # target variables are not balanced.
    # 1               16818
    # 0                3753
    # Two method to fight against inbalanced data: upsampling/cost setsitive learning.
    
    # Do upsampling to reduce the inbalancity of dataset.
    # Minority Class: Loss customer
    # To do upsample, use SMOTE algorithm.
    # Use imbalanced-learn library:
    '''
    imbalanced-learn requires the following dependencies:
    Python (>= 3.7)
    NumPy (>= 1.14.6)
    SciPy (>= 1.1.0)
    Scikit-learn (>= 1.0.1) 
    '''
    from imblearn.over_sampling import SMOTE
    smote_oversample = SMOTE(random_state=42, sampling_strategy=0.5)
    X_oversampled, y_oversampled = smote_oversample.fit_resample(X, y)
    print('New target rate:')
    print(y_oversampled.value_counts())
    
    # train-validation split in stratified mode.
    from sklearn.model_selection import train_test_split
    # for normal dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y.values)
    # for resampled dataset
    X_train_oversampled, X_val_oversampled, y_train_oversampled, y_val_oversampled = train_test_split(X_oversampled, 
                                                                                                        y_oversampled, 
                                                                                                        test_size=0.30, 
                                                                                          random_state=42, stratify=y_oversampled.values)
    # normalize numeric data
    # use z-score normalisation
    def z_score(df):
        df_std = df.copy()
        for col in df_std.columns:
            if col in numeric_columns:
                df_std[col] = (df_std[col] - df_std[col].mean()) / df_std[col].std()
        return df_std
    
    X_train_std = z_score(X_train)
    X_val_std = z_score(X_val)
  
    # also apply to the oversampled data set.
    X_train_oversampled_std = z_score(X_train_oversampled)
    X_val_oversampled_std = z_score(X_val_oversampled)    
    
    # let's code performance metric
    from sklearn.metrics import confusion_matrix
    def evaluate(y, y_hat , model_name):
        # use balanced accuracy metric
        cm = confusion_matrix(y, y_hat)
        print('*'*20)
        print('Confusion Matrix for model {0}: '.format(model_name))
        print(cm)
        sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])  
        specificity = cm[1,1]/(cm[1,0]+cm[1,1])
        bac = (sensitivity+specificity)/2
        print('Specifity of model {0}: '.format(model_name))
        print(specificity)
        print('Sensitivity of model {0}: '.format(model_name))
        print(sensitivity)
        print('Balanced accuracy of model {0}: '.format(model_name))
        print(bac)
        print('*'*20)
        
    import warnings
    warnings.filterwarnings("ignore")
    
    ''' _____Logistic Regression_____ '''
    from sklearn.linear_model import LogisticRegression
    # Use Lr with original dataset
    lr = LogisticRegression(random_state=0).fit(X_train_std.values, y_train.values)
    y_hat = lr.predict(X_val_std.values)
    evaluate(y_val.values, y_hat, model_name='Logistic Regression-Original')
    # Use Lr with oversampled dataset
    lr = LogisticRegression(random_state=0).fit(X_train_oversampled_std.values, y_train_oversampled.values)
    y_hat = lr.predict(X_val_oversampled_std.values)
    evaluate(y_val_oversampled.values, y_hat, model_name='Logistic Regression-Oversampled')
    # use lr with weight
    lr = LogisticRegression(random_state=0, class_weight='balanced').fit(X_train_std.values, y_train.values)
    y_hat = lr.predict(X_val_std.values)
    evaluate(y_val.values, y_hat, model_name='Logistic Regression-Weighted')
    
    
    ''' _____Decision Tree_____ '''
    from sklearn import tree
    # Use Dt with original dataset
    dt = tree.DecisionTreeClassifier(random_state=0, max_depth=5).fit(X_train_std.values, y_train.values)
    y_hat = dt.predict(X_val_std.values)
    evaluate(y_val.values, y_hat, model_name='Decision Tree-Original') # sensivity is too low.
    # Use Dt with oversampled dataset
    dt = tree.DecisionTreeClassifier(random_state=0, max_depth=2).fit(X_train_oversampled_std.values, y_train_oversampled.values)
    y_hat = dt.predict(X_val_oversampled_std.values)
    evaluate(y_val_oversampled.values, y_hat, model_name='Decision Tree-Oversampled')
    
    ''' _____SVM_____ '''
    
    from sklearn.svm import SVC
    # with original data set
    svm = SVC(gamma='auto', kernel='rbf', random_state=0).fit(X_train_std.values, y_train.values)
    y_hat = svm.predict(X_val_std.values)
    evaluate(y_val.values, y_hat, model_name='SVM-Original') 
    
    # with oversampled data set
    svm = SVC(gamma='auto', kernel='rbf', random_state=0).fit(X_train_oversampled_std.values, y_train_oversampled.values)
    y_hat = svm.predict(X_val_oversampled_std.values)
    evaluate(y_val_oversampled.values, y_hat, model_name='SVM-Oversampled') 
    
    # with weighted svm
    svm = SVC(gamma='auto', kernel='rbf', random_state=0, class_weight='balanced').fit(X_train_std.values, y_train.values)
    y_hat = svm.predict(X_val_std.values)
    evaluate(y_val.values, y_hat, model_name='SVM-Weighted') 
    
    # NOW LETS TRY ENSEMBLE METHODS
    
    ''' _____Random Forest_____ '''
    from sklearn.ensemble import RandomForestClassifier
    # Use rf with original dataset
    rf = RandomForestClassifier(random_state=0).fit(X_train_std.values, y_train.values)
    y_hat = rf.predict(X_val_std.values)
    evaluate(y_val.values, y_hat, model_name='Random Forest-Original')
    # Use rf with oversampled dataset
    rf = RandomForestClassifier(random_state=0).fit(X_train_oversampled_std.values, y_train_oversampled.values)
    y_hat = rf.predict(X_val_oversampled_std.values)
    evaluate(y_val_oversampled.values, y_hat, model_name='Random Forest-Oversampled')
    # use wighted rf

    rf_weighted = RandomForestClassifier(n_estimators=300,max_depth=5,
                                         criterion='gini',random_state=0, 
                                         class_weight='balanced').fit(X_train_std.values, y_train.values)
    y_hat = rf_weighted.predict(X_val_std.values)
    evaluate(y_val.values, y_hat, model_name='Random Forest-Weighted')
    

    ''' _____Ada Boost_____ '''
    from sklearn.ensemble import AdaBoostClassifier
    # Use adaboost with original dataset
    ab = AdaBoostClassifier(random_state=0, n_estimators=20, learning_rate=1.0).fit(X_train_std.values, y_train.values)
    y_hat = ab.predict(X_val_std.values)
    evaluate(y_val.values, y_hat, model_name='AdaBoost-Original')
    # Use ab with oversampled dataset
    ab = AdaBoostClassifier(random_state=0).fit(X_train_oversampled_std.values, y_train_oversampled.values)
    y_hat = ab.predict(X_val_oversampled_std.values)
    evaluate(y_val_oversampled.values, y_hat, model_name='AdaBoost-Oversampled')
    
  
    ''' _____XG Boost_____ '''
    from xgboost import XGBClassifier
    # original dataset
    xg_reg = XGBClassifier(objective ='binary:logistic', 
                              learning_rate = 0.1, max_depth = 5, 
                              alpha = 10, n_estimators = 10)
    xg_reg.fit(X_train_std.values, y_train.values)
    
    y_hat = xg_reg.predict(X_val_std.values)
    evaluate(y_val.values, y_hat, model_name='XBBoost-Original')
    
    
    # oversampled dataset
    xg_reg = XGBClassifier(objective ='binary:logistic', 
                              learning_rate = 0.2, max_depth = 10, 
                              alpha = 2, n_estimators = 20)
    xg_reg.fit(X_train_oversampled_std.values, y_train_oversampled.values)
    
    y_hat = xg_reg.predict(X_val_oversampled_std.values)
    evaluate(y_val_oversampled.values, y_hat, model_name='XBBoost-Oversampled')
    

    # weighted xgboost
    total_negative_examples = y_train.value_counts()[0]
    total_positive_examples = y_train.value_counts()[1]
    scale_pos_weight = total_negative_examples / total_positive_examples
    xg_reg_weighted = XGBClassifier(objective ='binary:logistic', 
                              learning_rate = 0.1, max_depth = 15, 
                              alpha = 20, n_estimators = 20, scale_pos_weight=scale_pos_weight)
    xg_reg_weighted.fit(X_train_std.values, y_train.values)
    
    y_hat = xg_reg_weighted.predict(X_val_std.values)
    evaluate(y_val.values, y_hat, model_name='XBBoost-Weighted')
    

    ''' 4) _____________________DEPLOYMENTS_____________________ '''
    # Candidate Models --> XBBOOST WEIGHTED, RANDOM FOREST WEIGHTED
    # precict test data
    
    test_df = test_data.df.copy()
    test_df.reset_index(inplace=True, drop=True)
    
    # preprocess test_df
    
    # first make every year format same
    test_df['CREATION_YEAR'] = test_df['CREATION_YEAR'].apply(lambda x: x.replace('-','.') if x is not np.nan else x)
    test_df['CREATION_YEAR'] = test_df['CREATION_YEAR'].apply(lambda x: x.replace('/','.') if x is not np.nan else x)
    test_df['CREATION_YEAR'] = test_df['CREATION_YEAR'].apply(lambda x: int(x.split('.')[-1]) if x is not np.nan else x)
    

    for idx, (year, currency, rev, rev_1, rev_2) in enumerate(test_df[['CREATION_YEAR','CURRENCY',
                                                          'REV_CURRENT_YEAR','REV_CURRENT_YEAR.1', 
                                                          'REV_CURRENT_YEAR.2']].to_numpy()):
        if currency=='Euro' or pd.isna(currency) or pd.isna(year) or pd.isna(rev) or pd.isna(rev_1) or pd.isna(rev_2):
            continue

        currency = currency_mapper[currency]
        converted_rev = c.convert(rev, currency, date=date(int(year), 1, 1))
        converted_rev1 = c.convert(rev_1, currency, date=date(int(year), 1, 1))
        converted_rev2 = c.convert(rev_2, currency, date=date(int(year), 1, 1))

        
        test_df.at[idx, 'REV_CURRENT_YEAR'] = converted_rev
        test_df.at[idx, 'REV_CURRENT_YEAR.1'] = converted_rev1
        test_df.at[idx, 'REV_CURRENT_YEAR.2'] = converted_rev2
    # drop uncessassry columns
    drop_cols = ['MO_ID', 'SO_ID','SALES_LOCATION', 'CUSTOMER', 'END_CUSTOMER','REV_CURRENT_YEAR.1','SO_CREATED_DATE']
    test_df.drop(columns=drop_cols, axis=1, inplace=True)
    test_df['CUSTOMER_GROWTH'] = extract_growth(test_df)
    # feature 2: profit
    test_df['PROFIT'] = extract_profit(test_df)

    
    # delete some cols
    test_df.drop(columns=['CREATION_YEAR', 'MO_CREATED_DATE'], axis=1, inplace=True)
    
    # feature 4: Which one is costly- Meterial or service?
    test_df['SERVICE_COSTLY'] = list(map(lambda x: 1 if x>0 else 0, list(test_df['SERVICE_COST'] - test_df['MATERIAL_COST'])))
    test_df.drop(columns=['SERVICE_LIST_PRICE', 'MATERIAL_COST', 'SERVICE_COST'], axis=1, inplace=True)
    cols = ['PRICE_LIST', 'ISIC', 'TECH', 'OFFER_TYPE','BUSINESS_TYPE','OWNERSHIP', 'CURRENCY']
    
    # fill na values for test set.

    groupped_data = data_backup_to_fill_na.groupby(['BUSINESS_TYPE']).median()
    groupped_data = groupped_data[['REV_CURRENT_YEAR', 'REV_CURRENT_YEAR.2', 'CUSTOMER_GROWTH']]
    groupped_data.reset_index(level=0, inplace=True)
    

    for idx, (business_type, rev_1, rev_2, growth) in enumerate(test_df[['BUSINESS_TYPE','REV_CURRENT_YEAR', 
                                                        'REV_CURRENT_YEAR.2', 'CUSTOMER_GROWTH']].to_numpy()):

        filtered_df = groupped_data[groupped_data['BUSINESS_TYPE']==business_type]
        if pd.isna(rev_1):
            test_df.at[idx, 'REV_CURRENT_YEAR'] = filtered_df['REV_CURRENT_YEAR']
        if pd.isna(rev_2):
            test_df.at[idx, 'REV_CURRENT_YEAR.2'] = filtered_df['REV_CURRENT_YEAR.2']
        if pd.isna(growth):
            test_df.at[idx, 'CUSTOMER_GROWTH'] = filtered_df['CUSTOMER_GROWTH']
   
    # fill ownership and Currency    
    test_df.fillna(method='ffill',inplace=True)
  
   
    test_df = pd.get_dummies(test_df, columns=cols)
    
    # binerize label
    test_df['COUNTRY'] = lb.fit_transform(test_df['COUNTRY'])
    
    # drop offer status
    test_set_id = test_df['TEST_SET_ID'].apply(lambda x: int(x))
    
    test_df.drop(['OFFER_STATUS', 'TEST_SET_ID'], axis=1, inplace=True)
    
    remove = ['OWNERSHIP_Individual Person', 'BUSINESS_TYPE_S', 'OFFER_TYPE_PAT', 
              'OFFER_TYPE_EN', 'OFFER_TYPE_XCPS', 'BUSINESS_TYPE_F', 
              'BUSINESS_TYPE_R', 'TECH_EPS','CUSTOMER_HOW_LONG']
    for k in remove:
        cols_to_delete.remove(k)
    
    test_df.drop(cols_to_delete, axis=1, inplace=True) # delete columns which not gonna used.
    

    test_df_std = z_score(test_df)
    # make prediction
    pred_xbboost = pd.Series(xg_reg_weighted.predict(test_df_std))
    pred_randomforest = pd.Series(rf_weighted.predict(test_df_std))
    
    df_entry = {'id':test_set_id,'prediction':pred_xbboost}
    submission_df = pd.DataFrame(df_entry)
    submission_df = submission_df.sort_values(by='id')
    submission_df.to_csv('predictions_melburmer.csv', index=False)
    
    
    
    
    
    
    
    
    
    

    
 
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


 
    
    
    
    
    
    
    
    
    



    
  