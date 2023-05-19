import os
import pandas as pd
from OneMax import OneMax
from FourPeaks import FourPeaks
from Knapsack import Knapsack
from TravellingSales import TravellingSales
from FlipFlop import FlipFlop
from Queens import Queens
from MaxKColor import MaxKColor
from NeuralNetworks import NeuralNetworks

DATA_FOLDER = './data'
FILENAME_1 = 'online_shoppers_intention.csv'
CATEGORICAL_COLUMNS_1 = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
Y_COLUMN_1 = 'Revenue'
IDENTIFIER_1 = 1

def loadData_1(encode_category = False):
    fullFilename = os.path.join(DATA_FOLDER, FILENAME_1)
    df = pd.read_csv(fullFilename)
    df.head()

    global NUMERIC_COLUMNS
    NUMERIC_COLUMNS = df.columns.difference(CATEGORICAL_COLUMNS_1)
    NUMERIC_COLUMNS = NUMERIC_COLUMNS.drop(Y_COLUMN_1)

    if encode_category:
        df_oneHot = df[CATEGORICAL_COLUMNS_1]
        df_oneHot = pd.get_dummies(df_oneHot, drop_first=True)
        df_droppedOneHot = df.drop(CATEGORICAL_COLUMNS_1, axis=1)
        return pd.concat([df_oneHot, df_droppedOneHot], axis=1)
    else:
        return df


def main():


    experiment = FlipFlop()
    experiment.run()

    # experiment = Knapsack()
    # experiment.run()

    # experiment = Queens()
    # experiment.run()


    # data = loadData_1(encode_category = True)
    # nn = NeuralNetworks()
    # nn.analyze(data)

    pass

if __name__ == "__main__":
    main()