
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import Datatransformation
import os
from src.components.model_trainer import ModelTrainer





if __name__=="__main__":
    obj=DataIngestion()
    #train_data_path,test_data_path=obj.initiate_data_ingestion()
    train_data_path=os.path.join(os.getcwd(),"artifacts","train_data.csv")
    test_data_path=os.path.join(os.getcwd(),"artifacts","test_data.csv")
    data_transformation = Datatransformation()
    train_tokenized,val_tokenized,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_tokenized,val_tokenized)

    