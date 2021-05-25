import csv
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

data = pd.read_csv('diamonds_gia_natural_round.csv')
# data = data.where(data['type'] == 'natural')
# data = data.where(data['report'] == 'GIA')
# data = data.where(data['shape'] == 'Round')

# model_name = 'price_per_carat ~ carat'
model_name = 'price ~ carat + color + clarity + cut'
model = ols(model_name, data=data).fit()

print(model.summary())
