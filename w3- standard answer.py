'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float/double/large


def get_entropy_of_dataset(df):
    entropy = 0
    for x in set(df.iloc[:, -1]):
        p = ((df.iloc[:, -1] == x).sum())/df.shape[0]
        if p != 0:
            entropy = entropy - (p * np.log2(p))
    return entropy


'''Return entropy of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float/double/large


def get_avg_info_of_attribute(df, attribute):
    entropy_of_attribute = 0
    for x in set(df[attribute]):
        df_x = df[df[attribute] == x]
        p_x = df_x.shape[0]/df.shape[0]
        E = get_entropy_of_dataset(df_x)
        entropy_of_attribute += (p_x * E)
    return abs(entropy_of_attribute)


'''Return Information Gain of the attribute provided as parameter'''
# input:int/float/double/large,int/float/double/large
# output:int/float/double/large


def get_information_gain(df, attribute):
    information_gain = 0
    information_gain = get_entropy_of_dataset(
        df) - get_entropy_of_attribute(df, attribute)
    return information_gain


''' Returns Attribute with highest info gain'''
#input: pandas_dataframe
#output: ({dict},'str')


def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    information_gains = {}
    selected_column = ''

    for x in df.columns[:-1]:
        information_gains[x] = get_information_gain(df, x)
        if(selected_column == '' or information_gains[selected_column] < information_gains[x]):
            selected_column = x

    return (information_gains, selected_column)

