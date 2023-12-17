import datetime
import json
import numpy as np
from sklearn import covariance, cluster
import yfinance as yf

# Завантаження прив'язок символів компаній до їх повних назв
input_file = 'company_symbol_mapping.json'
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())
symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань
start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)
quotes = []

for symbol in symbols:
    try:
        quote = yf.download(symbols[1], start=start_date, end=end_date, progress=False)
        quotes.append(quote)
    except:
        continue

# Вилучення котирувань, що відповідають
# відкриттю та закриттю біржі
opening_quotes = np.array([quote['Open'] for quote in quotes]).astype(float)
closing_quotes = np.array([quote['Close'] for quote in quotes]).astype(float)

# Обчислення різниці між двома видами котирувань
quotes_diff = closing_quotes - opening_quotes

# Нормалізація даних
X = quotes_diff.copy().T
X /= X.std(axis=0)

# Створення моделі графа
edge_model = covariance.GraphicalLassoCV()
# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

for i in range(num_labels + 1):
    print('Cluster', i + 1, '==>', ', '.join(names[labels == i]))