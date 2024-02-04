import sys,os
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import Save_obj
from src.utils import Dataset
from transformers import DataCollatorForSeq2Seq , AutoModelForSeq2SeqLM,TrainingArguments , Trainer
from src.utils import load_obj



@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join('artifacts','Bart_india_news')
    Bert_tokenizer_path= os.path.join('artifacts','Bert_tokenizer')
    model_ckpt="facebook/bart-large-cnn"
class ModelTrainer:
    def __init__(self):
        self.model_traner_config=ModelTrainingConfig()

    def initiate_model_training(self,train_tokenized,val_tokenized):
        try:
            logging.info("spliting dependent and independent verials from train and test data")
            train_dataset = Dataset(train_tokenized)
            val_dataset = Dataset(val_tokenized)
            model=AutoModelForSeq2SeqLM.from_pretrained(self.model_traner_config.model_ckpt)
            tokenizer=load_obj(self.model_traner_config.Bert_tokenizer_path)
            data_collator=DataCollatorForSeq2Seq(tokenizer,model=model)

            training_args=TrainingArguments(output_dir='bart_custom',
                                            num_train_epochs=1,
                                            warmup_steps=500,
                                            per_device_train_batch_size=4,
                                            per_device_eval_batch_size=4,
                                            weight_decay=0.01,
                                            logging_steps=10,
                                            eval_steps=500,
                                            save_steps=1e6,
                                            gradient_accumulation_steps=16
                                            )
            trainer = Trainer(model=model,args=training_args,
                  tokenizer=tokenizer,data_collator=data_collator,
                  train_dataset=train_dataset,eval_dataset=val_dataset)
            
            trainer.train()


            Save_obj(
                 file_path=self.model_traner_config.trained_model_file_path,
                 obj=trainer
            )
            logging.info("Model trained and saved successfully")
        except Exception as e:
            logging.info("error occcerd during model trainning")
            raise CustomException(e,sys)