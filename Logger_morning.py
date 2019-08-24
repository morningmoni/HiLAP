import logging


def myLogger(name=None, log_path='./tmp.log'):
    logger = logging.getLogger(name)
    if len(logger.handlers) != 0:
        print(f'[Logger_morning.py]reuse logger:{name}')
        return logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d][%(funcName)s] %(levelname)s-> %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S', filename=log_path, filemode='a')
    # define a new Handler to log to console as well
    console = logging.StreamHandler()
    # set a format which is the same for console use
    formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d][%(funcName)s] %(levelname)s-> %(message)s',
                                  datefmt='%a %d %b %Y %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logger.addHandler(console)
    logger.info(f'create new logger:{name}')
    logger.info(f'saving log to {log_path}')
    return logger


if __name__ == '__main__':
    logger = myLogger('111', log_path='runs/')
    logger.setLevel(10)
    logger.info('This is info message')
    logger.warning('This is warning message')

    logger = myLogger('111')
    logger.setLevel(10)
    logger.info('T2')
    logger.warning('This2')
