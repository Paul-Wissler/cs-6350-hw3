from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import Perceptron as perc


def q2a():
    print('STANDARD PERCEPTRON')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    
    model = perc.PerceptronModel(X, y, random_seed=False, rate=.1)
    error = 1 - model.test(X, y)
    print('TRAIN ERROR: ', error)
    error = 1 - model.test(X_test, y_test)
    print('TEST ERROR: ', error)
    print('')
    print(model.weights)
    print('\n')

    # STANDARD PERCEPTRON
    # TRAIN ERROR:  0.04243119266055051
    # TEST ERROR:  0.050000000000000044

    # WaveletVariance   -5.777450
    # WaveletSkew       -3.204904
    # WaveletCurtosis   -4.753204
    # ImageEntropy      -0.603608
    # MODEL_BIAS        -5.300000
    # dtype: float64


def q2b():
    print('VOTED PERCEPTRON')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    
    model = perc.VotedPerceptronModel(X, y, random_seed=False, rate=.1)
    # print(len(model.weights_list))
    weights_df = pd.DataFrame(model.weights_list)
    w_cols = list(weights_df.columns)
    weights_df['VOTES'] = model.votes_list
    weights_df[w_cols].plot()
    plt.ylabel('Weights')
    plt.xlabel('Update #')
    plt.savefig(Path('Instructions', 'q2b_weights_convergence.png'))
    plt.close()
    
    weights_df.VOTES.plot()
    plt.ylabel('Vote (Consecutive Rounds Without Failure)')
    plt.xlabel('Update #')
    plt.savefig(Path('Instructions', 'q2b_votes.png'))
    plt.close()
    
    # print(len(model.votes_list))
    error = 1 - model.test(X, y)
    print('TRAIN ERROR: ', error)
    error = 1 - model.test(X_test, y_test)
    print('TEST ERROR: ', error)
    print('\n')

    # VOTED PERCEPTRON
    # TRAIN ERROR:  0.012614678899082521
    # TEST ERROR:  0.014000000000000012


def q2c():
    print('AVERAGE PERCEPTRON')
    X, y = load_bank_note_data('train.csv')
    X_test, y_test = load_bank_note_data('test.csv')
    
    model = perc.AveragePerceptronModel(X, y, random_seed=False, rate=.1)
    error = 1 - model.test(X, y)
    print('TRAIN ERROR: ', error)
    error = 1 - model.test(X_test, y_test)
    print('TEST ERROR: ', error)
    print('')
    print(model.averaged_weights)
    print('\n')

    # AVERAGE PERCEPTRON
    # TRAIN ERROR:  0.014908256880733939
    # TEST ERROR:  0.014000000000000012

    # WaveletVariance   -37108.434439
    # WaveletSkew       -25185.487162
    # WaveletCurtosis   -25383.376175
    # ImageEntropy       -7638.087261
    # MODEL_BIAS        -34243.400000


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
