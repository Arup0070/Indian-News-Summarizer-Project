import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import MongoDBcoll
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd

@dataclass
class DataIngestionConfig:
    raw_data_path:str =os.path.join('artifacts','Eng_train.csv')
    train_data_path:str = os.path.join('artifacts','train_data.csv')
    test_data_path:str= os.path.join('artifacts','test_data.csv')
class DataIngestion:

    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        
        logging.info("Dataingestion procedure Started")
        try:
             
             df = MongoDBcoll()
             os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

             df.to_csv(self.ingestion_config.raw_data_path,index=False)
             train_set,test_set=train_test_split(df,test_size=0.20,random_state=42)

             train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True )
             test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True )

             logging.info("ingestion of data complete")
             logging.info(f"{train_set.head()}\n \n {test_set.head()}")
             logging.info('Ingestion of Data is completed')

             return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            
             )
  

   
        except Exception as e:
             logging.info("Exception occures at data ingestion stage")
             raise CustomException(e,sys)
        
