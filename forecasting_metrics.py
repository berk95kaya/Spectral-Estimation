# Adapted from https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
import numpy as np

EPSILON = 1e-10


def _error(actual, predicted):
    """ Simple error """
    return actual - predicted


def _naive_forecasting(actual, seasonality = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def _relative_error(actual, predicted, benchmark = None):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) /\
               (_error(actual[seasonality:], _naive_forecasting(actual, seasonality)) + EPSILON)

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)



def mrae(actual, predicted, benchmark= None):
    """ Mean Relative Absolute Error """
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))




METRICS = {
    
    'mrae': mrae,
   
}

