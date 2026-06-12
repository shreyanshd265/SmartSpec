from src.logger import logging
import sys
import sys
def get_error_message(error_message,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="error occured in the python typescript name [{0}] , in the line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error_message))
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail):
        super().__init__(error_message)
        self.error_message=get_error_message(error_message,error_detail)
    def __str__(self):
        return self.error_message
    


