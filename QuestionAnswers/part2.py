from pathlib import Path

import pandas as pd

import Perceptron as perc


def q2a():
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    
    model = perc.PerceptronModel(X, y, random_seed=False, rate=.1)
    # print(model.convergence_of_weights)
    error = 1 - model.test(X, y)
    print(error)
    error = 1 - model.test(X_test, y_test)
    print(error)


def q2b():
    print(perc.VotedPerceptronModel)


def q2c():
    print(perc.AveragePerceptronModel)


def load_bank_note_data(csv: str) -> (pd.DataFrame, pd.Series):
    X_cols = ['WaveletVariance', 'WaveletSkew', 'WaveletCurtosis', 'ImageEntropy']
    y_col = 'Label'

    train = load_data(csv)
    X = train[X_cols]
    y = encode_vals(train[y_col])
    return X, y


def load_data(csv: str) -> pd.DataFrame:
    return pd.read_csv(
        Path('bank-note', 'bank-note', csv),
        names=['WaveletVariance', 'WaveletSkew', 'WaveletCurtosis', 'ImageEntropy', 'Label']
    )


def encode_vals(y: pd.Series):
    return y.replace({0: -1})
