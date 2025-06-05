import sys
import logging
from src.logger import logging


def error_message_detail(error,error_detail: sys):
    _,_,exc_tb = error_detail.exc_info()
    error_message = "error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        exc_tb.tb_frame.f_code.co_filename,
        exc_tb.tb_lineno,
        str(error)
    )
    return error_message
    

class CustomException(Exception):
    def __init__(self , error_message, error_details: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_details)

    def __str__(self):
        return self.error_message
