import pandas as pd

from encoding_functions import define_number_of_outputs, get_weights_and_bias, encoding_model
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
    models = build_model(dict['heart_statlog'], 'heart_statlog')
    encoding_model(models['heart_statlog'][0], 'heart_statlog')
    #res = define_number_of_outputs(models['hepatitis'][0])
    #get_weights_and_bias(models['hepatitis'][0])
    #print(len(dict['hepatitis'].columns))
    #print(models['hepatitis'][0].weights)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
