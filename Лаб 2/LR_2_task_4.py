import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Читання даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line[:-1].split(', ')
        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1
        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])
X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Завантажуємо алгоритми моделі
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='scale', C=1.0)))

# Розділення датасету на навчальну та контрольну вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Оцінюємо модель на кожній ітерації
names = []
model_metrics = {}
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    accuracy_scorer = make_scorer(accuracy_score)
    f1_scorer = make_scorer(f1_score)
    recall_scorer = make_scorer(recall_score)
    precision_scorer = make_scorer(precision_score)

    # Оцінка якості моделі цієї ітерації
    accuracy_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=accuracy_scorer)
    f1_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=f1_scorer)
    recall_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=recall_scorer)
    precision_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=precision_scorer)

    model_metrics[name] = {
        'Accuracy': accuracy_scores,
        'F1 Score': f1_scores,
        'Recall': recall_scores,
        'Precision': precision_scores
    }
    names.append(name)

    print(f'{name}:')
    print(f'Accuracy: {accuracy_scores.mean():.5f} ({accuracy_scores.std():.5f})')
    print(f'F1 Score: {f1_scores.mean():.5f} ({f1_scores.std():.5f})')
    print(f'Recall: {recall_scores.mean():.5f} ({recall_scores.std():.5f})')
    print(f'Precision: {precision_scores.mean():.5f} ({precision_scores.std():.5f})\n')

# Порівняння алгоритмів
accuracy_values = [model_metrics[name]['Accuracy'] for name in names]
f1_score_values = [model_metrics[name]['F1 Score'] for name in names]
recall_values = [model_metrics[name]['Recall'] for name in names]
precision_values = [model_metrics[name]['Precision'] for name in names]

# Графіки якості моделей
pyplot.boxplot(accuracy_values, labels=names)
pyplot.title('Accuracy Comparison')
pyplot.show()

pyplot.boxplot(f1_score_values, labels=names)
pyplot.title('F1 score Comparison')
pyplot.show()

pyplot.boxplot(recall_values, labels=names)
pyplot.title('Recall Comparison')
pyplot.show()

pyplot.boxplot(precision_values, labels=names)
pyplot.title('Precision Comparison')
pyplot.show()