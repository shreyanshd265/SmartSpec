import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import sys
import os
from src.utils import load_object
from src.components.data_transformation import cleaning_function

class PredictPipleine:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('model.pkl')
            preprocessor_path=os.path.join('preprocessor.pkl')
            preprocessor=load_object(file_path=preprocessor_path)
            model=load_object(file_path=model_path)
            features = cleaning_function(features)
            data_scaled=preprocessor.transform(features)
            predicted_value=model.predict(data_scaled)
            predicted_value=np.exp(predicted_value)
            return predicted_value
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 Rating:float,
                 Display:str,
                 Generation:str,
                 Core:str,
                 Ram:str,
                 SSD:str,
                 Warranty:str,
                 Model:str,
                 OS:str,
                 Graphics:str):
        self.Rating=Rating
        self.Display=Display
        self.Generation=Generation
        self.Core=Core
        self.Ram=Ram
        self.SSD=SSD
        self.Warranty=Warranty
        self.Model=Model
        self.OS=OS
        self.Graphics=Graphics

    def get_dataframe(self):
        try:
            feature_dic={
                "Rating":[self.Rating],
                "Display":[self.Display],
                "Generation":[self.Generation],
                "Core":[self.Core],
                "Ram":[self.Ram],
                "SSD":[self.SSD],
                "Warranty":[self.Warranty],
                "Model":[self.Model],
                "OS":[self.OS],
                "Graphics":[self.Graphics]
            }
            return pd.DataFrame(feature_dic)
        except Exception as e:
            raise CustomException(e,sys)
