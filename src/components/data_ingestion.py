import os
from src.exception import CustomException
import pandas as pd
from src.logger import logging
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
from sklearn.model_selection import GridSearchCV

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts',"train.csv")
    test_data_path=os.path.join('artifacts',"test.csv")
    raw_data_path=os.path.join('artifacts',"raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_ingestion(self):
        logging.info("data ingestion started")
        try:
            df=pd.read_csv('NOTEBOOK/laptop.csv')
            logging.info("data read successfully")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("train test split initiated")
            df_train,df_test=train_test_split(df,random_state=42,test_size=0.25)
            df_train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            df_test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("train test data saved successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        




if __name__=="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_ingestion()
    data_transform=DataTransformation()
    train_arr,test_arr,preprocessor_obj_file_path=data_transform.initiate_data_transformation(train_path=train_path,test_path=test_path)
    model=ModelTrainer()
    r2= model.initiate_model_training(train_arr=train_arr,test_arr=test_arr)
    print(r2)
    
