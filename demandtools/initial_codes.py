# This file imports 'estimation_data.txt' to use during development. This file, as well as the 'estimation_data.txt'
# are not actual parts of this package and will be removed before the final version.

import numpy as np
import pandas as pd

data = pd.read_csv("./demandtools/train.csv")
#price = np.log(data.loc[:, data.columns.str.startswith('pt_')])
price = data.loc[:, data.columns.str.startswith('pt_')]
#expenditure = np.log(data.loc[:, 'total_expenditure'])
expenditure = data.loc[:, 'total_expenditure']
demographics = data.loc[:, data.columns.str.startswith('d_')]
#demographics = demographics.drop("d_hh_size", axis=1)
budget_share = data.loc[:, data.columns.str.startswith('wt_')]



###Initial coefficients
alpha = np.random.randn(10)
gamma = np.random.randn(10, 10)
gamma = (gamma + gamma.T - np.diag(gamma.diagonal()))*0.5
alpha0 = 0
alpha_demographic = np.random.randn(10, 3)
beta = np.random.randn(10)
