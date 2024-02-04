from src.exception import CustomException
from src.logger import logging
from transformers import AutoTokenizer
import torch                               
from dataclasses import dataclass
import os,sys
import pandas as pd
from src.utils import Save_obj
import numpy as np

@dataclass
class Datatransformationconfig:
    preprocessor_file_obj_path = os.path.join('artifacts','Bert_tokenizer')

class Datatransformation:

    def __init__(self):
        self.data_transformation_config=Datatransformationconfig()
        
    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation process Started")
            model_ckpt="facebook/bart-large-cnn"
            tokenizer=AutoTokenizer.from_pretrained(model_ckpt)

            return tokenizer

        except Exception as e:
            logging.info("Error in data transformation module")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Data Traformation started")
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)
            logging.info("Train test Data read from file successfull")

            tokenizer=self.get_data_transformation_obj()
            X_train= list(train_data["dialogue"])
            y_train= list(train_data["summary"])
            X_val= list(train_data["dialogue"])
            y_val= list(train_data["summary"])

            logging.info("Data separation done for input and target feature and preprocessing Started ")

            train_tokenized = tokenizer(X_train,text_target=y_train,
                                        max_length=1024,truncation=True)
            val_tokenized = tokenizer(X_val,text_target=y_val,
                                        max_length=1024,truncation=True)

            
            Save_obj(
                file_path=self.data_transformation_config.preprocessor_file_obj_path,
                obj=tokenizer
            )
            logging.info("Bert_tokenizer Model saved")
            logging.info("Data transformation done successfully")
            
            return(
                train_tokenized,
                val_tokenized,
                self.data_transformation_config.preprocessor_file_obj_path
            
            )
        
            

        except Exception as e:
            logging.info("error in data transformation module")
            raise CustomException(e,sys)







