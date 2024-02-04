from src.logger import logging
from src.exception import CustomException
import os,sys
from src.utils import load_obj
import json
from transformers import pipeline
from transformers import MBartForConditionalGeneration,MBart50TokenizerFast

class PredictPipeline:
    def __init__(self) -> None:
        pass
    def load_models():
        try:
        
            summary_model_path=os.path.join('artifacts','Bart_india_news')
            translate_eng_tokenizer_path=os.path.join("artifacts","eng_tokenizer") 
            text_translation_model_path=os.path.join("artifacts","mbart-large-50/")
            text_translation_model = MBartForConditionalGeneration.from_pretrained(text_translation_model_path)
            news_summary_model=pipeline('summarization',model=summary_model_path)
            eng_tokenizer=MBart50TokenizerFast.from_pretrained(translate_eng_tokenizer_path,src_lang="en_XX")
            
            return text_translation_model,news_summary_model,eng_tokenizer 
    
            
        except Exception as e:
            logging.info("Error in model loading from local process")
            raise CustomException(e,sys)
    def other_to_eng(text,language,text_translation_model):
        try:
            translate_input_tokenizer=MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt",src_lang=language)
            model_inputs=translate_input_tokenizer(text,return_tensors='pt')
            genarated_tokens=text_translation_model.generate(**model_inputs,
                                            forced_bos_token_id=translate_input_tokenizer.lang_code_to_id["en_XX"]
                                       )
            translation=translate_input_tokenizer.batch_decode(genarated_tokens,skip_special_tokens=True)
            return translation
        except Exception as e:
            logging.info("Error in Other to english translation process")
            raise CustomException(e,sys)
    def summary_genaretor(translation,news_summary_model):
        try:
            eng_text_summary=news_summary_model(translation)[0]['summary_text']
            return eng_text_summary
        except Exception as e:
            logging.info("Error in Summary genaretion process")
            raise CustomException(e,sys)
    def eng_to_other(eng_text_summary,language,eng_tokenizer,text_translation_model):
        try:
            model_inputs=eng_tokenizer(eng_text_summary,return_tensors='pt')
            genarated_tokens=text_translation_model.generate(**model_inputs,
                                            forced_bos_token_id=eng_tokenizer.lang_code_to_id[language]
                                            )
            out_sum_text=eng_tokenizer.batch_decode(genarated_tokens,skip_special_tokens=True)

            return out_sum_text[0]

            
        except Exception as e:
            logging.info("Error in model prediction process")
            raise CustomException(e,sys)

