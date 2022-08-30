# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from neural_net_functions import build_model

def import_dataframe(dataframes):
    dictionary = {}

    for df_name in dataframes:
        df = pd.read_csv(".\datasets_files\\" + df_name, index_col=[0],  sep=',', na_values=[''])
        #print(df.describe().transpose())
        dictionary[df_name] = df

    return dictionary

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    with open(".\datasets_configuration\datasets_list") as f:
        dataframes = f.read().splitlines()

    dict = import_dataframe(dataframes)
    datasets = build_model(dict['hepatitis'], 'hepatitis')
    #print(len(dict['auto'].columns))
    #print(datasets['auto'][0].weights)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
