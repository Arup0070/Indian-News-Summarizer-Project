import os
import logging
from datetime import datetime

file_name=f"{datetime.now().strftime('%Y-%d-%m_%H-%M-%S')}.log"
file_path=os.path.join(os.getcwd(),"logs",file_name)
os.makedirs(file_path,exist_ok=True)
LOG_FILE_PATH=os.path.join(file_path,file_name)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

