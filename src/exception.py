import os
import sys
from src.logger import logging


def error_message_detail(error,error_detail:sys):
    _,_,exc_td = error_detail.exc_info()
    file_name=exc_td.tb_frame.f_code.co_filename

    error_message = ("Error occured in python script name [{0}] line number [{1}] Error massage [{2}]".format(file_name,exc_td.tb_lineno,str(error)))

    return error_message

class CustomException(Exception):
    def __init__(self, error_massage,error_details:sys):
        super().__init__(error_massage)
        self.error_massage=error_message_detail(error_massage,error_detail=error_details)

    def __str__(self) -> str:
        return self.error_massage
    
'''
if __name__=="__main__":
    logging.info("Logging has started")

    try:
        a=1/0
    except Exception as e:
        logging.info('Dicision by zero') 
        raise CustomException(e,sys)
'''