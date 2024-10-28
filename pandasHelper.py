# coding=gbk
from source.utils.StringKeyUtils import StringKeyUtils
import pandas


class pandasHelper:
    """  pandas�ӿڷ�װ������ """

    INT_READ_FILE_WITHOUT_HEAD = None
    INT_READ_FILE_WITH_HEAD = 0
    STR_WRITE_STYLE_APPEND_NEW = 'a+'
    STR_WRITE_STYLE_APPEND = 'a'
    STR_WRITE_STYLE_WRITE_TRUNC = 'w'
    INT_WRITE_WITH_HEADER = True
    INT_WRITE_WITHOUT_HEADER = False

    @staticmethod
    def readTSVFile(fileName, header=INT_READ_FILE_WITHOUT_HEAD, sep=StringKeyUtils.STR_SPLIT_SEP_TSV, low_memory=True):  # ��һΪ�ޱ�ͷ
        train = pandas.read_csv(fileName, sep=sep, header=header, low_memory=low_memory)
        return train

    @staticmethod
    def toDataFrame(data, columns=None, dtype=None):
        return pandas.DataFrame(data, columns=columns, dtype=dtype)

    @staticmethod
    def writeTSVFile(fileName, dataFrame, writeStyle=STR_WRITE_STYLE_WRITE_TRUNC,
                     header=INT_WRITE_WITH_HEADER):
        """ д��tsv�ļ� ����header�ֶ�"""
        with open(fileName, writeStyle, encoding='utf-8') as write_tsv:
            print(fileName)
            write_tsv.write(dataFrame.to_csv(sep=StringKeyUtils.STR_SPLIT_SEP_TSV, index=False, header=header))