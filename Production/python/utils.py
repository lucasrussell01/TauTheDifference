import logging


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': "\033[0;37m",  # White
        'INFO': "\033[0;34m",   # Blue
        'WARNING': "\033[0;33m", # Yellow/Orange
        'ERROR': "\033[0;31m",   # Red
        'CRITICAL': "\033[1;31m", # Bold Red
        'RESET': "\033[0m",      # Reset color
    }

    def format(self, record):
        levelname_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{levelname_color}{record.levelname:<8}{self.COLORS['RESET']}"
        return super().format(record)

def get_logger(debug):
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(levelname)-8s: %(message)-140s [Line %(lineno)d]'))
    logger.addHandler(handler)
    return logger

