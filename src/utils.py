from src.logger import logging
from src.exception import CustomException
import pymongo
from pymongo import MongoClient
import pandas as pd
import sys
import os
import certifi
from transformers import AutoTokenizer
from transformers import MBartForConditionalGeneration,MBart50TokenizerFast
import torch

def MongoDBcoll():
    logging.info("Connecting to MongoDB database")
    try:
        ca= certifi.where()
        cluster = MongoClient("mongodb+srv://arup92327:Arup0070@cluster0.e9r83iz.mongodb.net/?retryWrites=true&w=majority",tlsCAFile=ca)
        db=cluster["NLP_DB"]
        coll=db["Indian_news_data"]
        li=[]
        for i in coll.find({},{"_id":0,"id":0}):
            li.append(i)
        Data=pd.DataFrame(li)
        logging.info(f"{Data.head()} \n Collected Data")
        cluster.close()
        return Data
    except Exception as e:
        logging.info("not able to collect data from MongoBD Data Base")
        raise CustomException(e,sys)

def Save_obj(file_path,obj):
    logging.info("file object Saveing process started")
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        obj.save_pretrained(file_path)
    except Exception as e:
        logging.info("Error in file obj save")
        raise CustomException(e,sys)
    
def load_obj(file_path):
    try:
        if str(file_path).split("\\")[-1] == "Bert_tokenizer":
            tokenizer=AutoTokenizer.from_pretrained(file_path)
        else:
            tokenizer=MBart50TokenizerFast.from_pretrained("artifacts/eng_tokenizer",src_lang="en_XX")
        return tokenizer
    except Exception as e:
        logging("error occured in Object load module")
        raise CustomException(e,sys) 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
          item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])   

'''
if __name__=="__main__":
    ran=RandomForestRegressor()
    preprocessor_file_obj_path = os.path.join('artifacts','preprocessor.pkl')
    data = Save_obj(file_path=preprocessor_file_obj_path,obj=ran)
'''