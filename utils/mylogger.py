import config as c
import logging
from logging import *

chatid2loggers = {}
def getLoggerByChatId(chat_id:int) -> logging.Logger:
    if chat_id in chatid2loggers:
        return chatid2loggers[chat_id]
    else:
        logger = logging.getLogger('MF_nlp_chatbot')
        hdlr = logging.FileHandler('logs/chatbot_%d.log' %chat_id)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
        chatid2loggers[chat_id] = logger
        return logger