#coding=gbk
from source.data.bean.Beanbase import BeanBase

class DataInterceptor:
    
    ''' ����������;���ʵ����ת������'''
    
    @staticmethod
    def convertFromBeanbaseToOutput(bean):
        if(not isinstance(bean, BeanBase)):
            return None #��������ת��ʧ��
        
    @staticmethod
    def convertFromOutputToBeanBase(bean):
        if(not isinstance(bean, BeanBase)):
            return None #��������ת��ʧ��
            