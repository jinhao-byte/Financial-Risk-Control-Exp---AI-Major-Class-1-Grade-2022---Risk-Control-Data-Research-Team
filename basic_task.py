import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored as cl
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 需要的模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# 读取并预处理数据
df = pd.read_csv('data/creditcard.csv')
df.drop('Time', axis = 1, inplace = True)

sc = StandardScaler()
df['Amount'] = sc.fit_transform(df['Amount'].values.reshape(-1, 1))

X = df.drop('Class', axis = 1).values
y = df['Class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 0
)

# 逻辑回归

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_yhat = lr.predict(X_test)

# 支持向量机
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
svm_yhat = svm.predict(X_test)

# 随机森林

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=4)
rf.fit(X_train, y_train)
rf_yhat = rf.predict(X_test)

# BP神经网络

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=300)
mlp.fit(X_train, y_train)
mlp_yhat = mlp.predict(X_test)

# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_yhat = nb.predict(X_test)

# 评估：Accuracy & F1
print(cl('ACCURACY SCORE', attrs=['bold']))
print(cl('Logistic Regression: {}'.format(accuracy_score(y_test, lr_yhat)), color='red'))
print(cl('SVM: {}'.format(accuracy_score(y_test, svm_yhat))))
print(cl('Random Forest: {}'.format(accuracy_score(y_test, rf_yhat))))
print(cl('Neural Network (MLP): {}'.format(accuracy_score(y_test, mlp_yhat)), color='green'))
print(cl('Naive Bayes: {}'.format(accuracy_score(y_test, nb_yhat))))

print(cl('\nF1 SCORE', attrs=['bold']))
print(cl('Logistic Regression: {}'.format(f1_score(y_test, lr_yhat)), color='red'))
print(cl('SVM: {}'.format(f1_score(y_test, svm_yhat))))
print(cl('Random Forest: {}'.format(f1_score(y_test, rf_yhat))))
print(cl('Neural Network (MLP): {}'.format(f1_score(y_test, mlp_yhat)), color='green'))
print(cl('Naive Bayes: {}'.format(f1_score(y_test, nb_yhat))))

# 混淆矩阵绘图函数
def plot_confusion_matrix(cm, classes, title, normalize=False, cmap=plt.cm.Blues):
    title = 'Confusion Matrix of {}'.format(title)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 绘制混淆矩阵
plt.rcParams['figure.figsize'] = (6, 6)
models = {
    'Logistic Regression': (lr, lr_yhat),
    'SVM': (svm, svm_yhat),
    'Random Forest': (rf, rf_yhat),
    'Neural Network': (mlp, mlp_yhat),
    'Naive Bayes': (nb, nb_yhat)
}

for name, (model, yhat) in models.items():
    cm = confusion_matrix(y_test, yhat, labels=[0, 1])
    plot_confusion_matrix(cm, ['Non-Fraud (0)', 'Fraud (1)'], title=name)
    plt.savefig(f'{name.replace(" ", "_").lower()}_cm.png')
    plt.show()
