import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path :
         os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(x_train,x_test,y_train,y_test,model,para):
    try:
        model_report={}
        for i in range(len(list(model))):
            model_name=list(model.keys())[i]
            param=para[model_name]
            model_obj=list(model.values())[i]
            gs=GridSearchCV(model_obj,param,cv=5)
            gs.fit(x_train,y_train)
            model_obj.set_params(**gs.best_params_)
            model_obj.fit(x_train,y_train)
            y_pred=model_obj.predict(x_test)
            r2=r2_score(y_test,y_pred)
            model_report[model_name]=r2
        return model_report
        
    except Exception as e:
        raise CustomException(e,sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:

        raise CustomException(e, sys)