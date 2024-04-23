import numpy as np
from scipy.stats import norm
import pandas as pd

## Data generating process with many nonlinearities
def dgp_nonlinear(n):
    # Set seed
    np.random.seed(3172024)
    datasize=n

    # Draw an error
    e = norm.rvs(size=(datasize,1), scale = .005)

    X1 = np.random.uniform(0, 2, (datasize, 1))
    X2 = norm.rvs(size=(datasize,1), scale = 1)
    X3 = X1 * np.sin(X2) + norm.rvs(size=(datasize,1), scale = .1)

    beta = np.random.normal(0, 1, 3)

    Y = np.dot(X1, beta[0]) + np.dot(X2, beta[1]) + np.dot(X3, beta[2]) + e

    df = pd.DataFrame(np.concatenate((Y, X1, X2, X3), axis=1), columns=['Y', 'X1', 'X2', 'X3'])
    return df
