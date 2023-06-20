import logging

from network.data_loader import data_copy
from network.train import train

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s]%(asctime)s %(filename)s %(lineno)d %(message)s',
                        datefmt='%Y %b %d %H:%M:%S', )
    # data_copy(r'Z:\network\dataset\\', r'D:\dataset\\')
    train(r'D:\dataset\\')