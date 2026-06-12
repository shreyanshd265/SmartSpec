import pandas as pd
import numpy as np
import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils import save_object


def cleaning_function(df):
        try:
            df=df.reset_index(drop=True)
            df.drop(columns=['Unnamed: 0'],inplace=True,errors='ignore')
            if pd.isna(df['Rating'].mean()):
                df['Rating']=df['Rating'].fillna(4.0)
            else :
                df['Rating']=df['Rating'].fillna(df['Rating'].mean())

            df['Brand']=df['Model'].str.split().str[0]
            df.drop(columns=['Model'],inplace=True,errors='ignore')
            top_brands=df['Brand'].value_counts()
            top_brands=top_brands[top_brands>20].index.tolist()
            df['Brand']=df['Brand'].apply(lambda x : x if x in top_brands else 'other')

            ## Display column EDA
            if not df['Display'].mode().empty:
                df['Display']=df['Display'].fillna(df['Display'].mode().iloc[0])
            else :
                df['Display']=df['Display'].fillna('15.6')
            df['Display']=df['Display'].str.extract(r'(\d+\.?\d*)')
            df['Display']=pd.to_numeric(df['Display'])
            invalid_display=[1.0,4.0]
            df['Display']=df['Display'].apply(lambda x : np.nan if x in invalid_display else x)
            if pd.isna(df['Display'].median()):
                df['Display']=df['Display'].fillna(15.6)
            else :
                df['Display']=df['Display'].fillna(df['Display'].median())

            ## price column EDA if exists
            if 'Price' in df.columns :
                df['Price']=df['Price'].str.replace('₹','',regex=False)
                df['Price']=df['Price'].str.replace(',','',regex=False)
                df['Price']=pd.to_numeric(df['Price'])
                df['Price']=np.log(df['Price'])

            ## Generation column EDA
            df['Generation']=df['Generation'].str.extract(r'(\d+)')
            df['Generation']=pd.to_numeric(df['Generation'])
            if pd.isna(df['Generation'].median()):
                df['Generation'] = df['Generation'].fillna(10)
            else:
                df['Generation'] = df['Generation'].fillna(df['Generation'].median())

            ##Core coluimn EDA
            core_map={'Dual': '2', 'Quad': '4', 'Hexa': '6', 'Octa': '8'}
            for word,num in core_map.items():
                df['Core']=df['Core'].str.replace(word,num,regex=False)
            df['Core']=df['Core'].str.extract(r'(\d+)')
            df['Core']=pd.to_numeric(df['Core'])
            df['Core']=df['Core'].apply(lambda x : np.nan if x>32 else x)
            if pd.isna(df['Core'].median()):
                df['Core']=df['Core'].fillna(4)
            else :
                df['Core']=df['Core'].fillna(df['Core'].median())

            ## RAM  column eda
            df['Ram']=df['Ram'].str.extract(r'(\d+)')
            df['Ram']=pd.to_numeric(df['Ram'])
            valid_ram=[2, 4, 8, 12, 16, 18, 32, 36, 48, 64]
            df['Ram']=df['Ram'].apply(lambda x : x if x in valid_ram else np.nan)
            if pd.isna(df['Ram'].median()):
                df['Ram'] = df['Ram'].fillna(8)
            else:
                df['Ram'] = df['Ram'].fillna(df['Ram'].median())

            ## SSD column EDA
            df.loc[df['SSD'] == '14 inches, 2160 x 1440 pixels', 'SSD'] = np.nan
            df.loc[df['SSD'] == '11.6 inches, 1366 x 768 pixels', 'SSD'] = np.nan
            df.loc[df['SSD'] == 'Intel Arc Graphics', 'SSD'] = np.nan
            df.loc[df['SSD'] == '8 GB NVIDIA GeForce RTX 4060', 'SSD'] = np.nan
            replace_lst = ['Operating System: Windows 11 Home', '16 GB NVIDIA GeForce RTX 4090', '8 GB NVIDIA GeForce RTX 4070']
            for val in replace_lst:
                df.loc[df['SSD'] == val, 'SSD'] = np.nan
            ssd_val = {'1 TB SSD': '1000', '2 TB SSD ': '2000', '4 TB SSD  ': '4000'}
            for word, num2 in ssd_val.items():
                df['SSD'] = df['SSD'].str.replace(word, num2, regex=False)
            df['SSD'] = df['SSD'].str.extract(r'(\d+)')
            df['SSD'] = pd.to_numeric(df['SSD'])
            valid_ssd = [128, 256, 512, 1000, 2000]
            df['SSD'] = df['SSD'].apply(lambda x: x if x in valid_ssd else np.nan)
            if pd.isna(df['SSD'].median()):
                df['SSD'] = df['SSD'].fillna(512)
            else:
                df['SSD'] = df['SSD'].fillna(df['SSD'].median())

            ##OS column EDA
            df['OS']=df['OS'].str.split().str[0]
            valid_os=['Windows','Mac']
            df['OS']=df['OS'].apply(lambda x : x if x in valid_os else np.nan)
            if not df['OS'].mode().empty :
                df['OS']=df['OS'].fillna(df['OS'].mode().iloc[0])
            else :
                df['OS']=df['OS'].fillna('Windows')

            ##Warranty column EDA
            valid_warranty=['1 Year Warranty','2 Year Warranty','3 Year Warranty']
            df['Warranty']=df['Warranty'].apply(lambda x : x if x in valid_warranty else np.nan)
            df['Warranty']=df['Warranty'].str.extract(r'(\d+)').astype(float)
            if pd.isna(df['Warranty'].median()):
                df['Warranty'] = df['Warranty'].fillna(1)
            else:
                df['Warranty'] = df['Warranty'].fillna(df['Warranty'].median())

            ##Graphics column EDA
            graphics_list=['nvidia','amd','intel','Apple']

            def categorize_graphics(g):
                g=str(g).lower()
                if 'apple' in g :
                    return 'Apple'
                if 'amd' in g:
                    return 'AMD'
                if 'intel' in g :
                    return 'Intel'
                if 'nvidia' in g:
                    return 'NVIDIA'
                else :
                    return np.nan
            
            df['Graphics']=df['Graphics'].apply(categorize_graphics)
            if not df['Graphics'].mode().empty:
                df['Graphics']=df['Graphics'].fillna(df['Graphics'].mode().iloc[0])
            else :
                df['Graphics']=df['Graphics'].fillna('Intel')
            return df
        except Exception as e:
            raise CustomException(e,sys)




@dataclass
class DataTransformationConfig():
    preprocessor_file_path=os.path.join('preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config=DataTransformationConfig()


    def get_preprocesoor_object(self):
       try:
            num_features=['Rating', 'Display', 'Generation', 'Core', 'Ram', 'SSD', 'Warranty']
            cat_features=['Brand','OS','Graphics']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("standardScalerization",StandardScaler())
                ]
                )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoding",OneHotEncoder())
                ]
            )
            logging.info("the pipleine made successfully and cat_feature and numerical features found successfully")
            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,num_features),
                    ("categorical_pipeline",cat_pipeline,cat_features)
                ]
            )
            return preprocessor
       except Exception as e:
           raise CustomException(e,sys)


   
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            train_df=cleaning_function(train_df)
            test_df=cleaning_function(test_df)

            

            train_df_input_features=train_df.drop(columns=['Price'])
            train_df_output_features=train_df['Price']
            test_df_input_features=test_df.drop(columns=['Price'])
            test_df_output_features=test_df['Price']

            logging.info("extracting the preprocessor object")
            preprocessor_obj=self.get_preprocesoor_object() 

            logging.info("applying the preprocessing to the train and test input features")
            input_feature_train_arr=preprocessor_obj.fit_transform(train_df_input_features)
            input_feature_test_arr=preprocessor_obj.transform(test_df_input_features)

            
            train_arr = np.c_[input_feature_train_arr, np.array(train_df_output_features)]
            test_arr = np.c_[input_feature_test_arr, np.array(test_df_output_features)]

            save_object(
                file_path=self.transformation_config.preprocessor_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_file_path
            )
        except Exception as e:
            raise CustomException(e,sys) 

       


        


    
         