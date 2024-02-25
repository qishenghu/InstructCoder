import logging
import colorama
from colorama import Fore, Style



class ColoredLoggingFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: Fore.LIGHTBLACK_EX + "{format}" + Style.RESET_ALL,
        logging.INFO: Fore.LIGHTBLACK_EX + "{format}" + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + "{format}" + Style.RESET_ALL,
        logging.ERROR: Fore.RED + "{format}" + Style.RESET_ALL,
        logging.CRITICAL: Fore.LIGHTRED_EX + "{format}" + Style.RESET_ALL,
    }

    def __init__(self, 
                 fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s" # (%(filename)s:%(lineno)d)
                ):
        # Initialize Colorama to work on Windows as well
        colorama.init()
        self.formatters = dict()
        for levelno in self.FORMATS.keys():
            log_fmt = self.FORMATS.get(levelno).format(format=fmt)
            self.formatters[levelno] = logging.Formatter(log_fmt)


    def format(self, record):
        return self.formatters.get(record.levelno).format(record)
    

# The convention is to call this at the top of submodule py files.
def init_logger(name):
    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create a console handler and set the custom formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredLoggingFormatter())

    # Add the console handler to the custom logger
    logger.addHandler(console_handler)

    return logger