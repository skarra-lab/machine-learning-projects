import sys
import numpy as np
import pandas as pd
import time
import pylab as pl
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report
import itertools
import matplotlib.pyplot as plt

transfusion_df = pd.read_csv("transfusion.data.txt")
print(transfusion_df)

transfusion_df = transfusion_df[["Recency (months)","Frequency (times)","Monetary (c.c. blood)","Time (months)","Donation in March 2007"]]
print(transfusion_df.columns)
transfusion_df_target = transfusion_df["Donation in March 2007"].astype('int')
print(transfusion_df.shape)

X = np.asarray(transfusion_df[["Recency (months)","Frequency (times)","Monetary (c.c. blood)","Time (months)"]])
y = np.asarray(transfusion_df['Donation in March 2007'])
print(X[0:5])
print(y[0:5])

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print("Train set: ", X_train.shape, y_train.shape)
print("Test set: ", X_test, y_test)

lr = LogisticRegression(C=0.01, solver="liblinear").fit(X_train, y_train)
print(lr)

pred_lr = lr.predict(X_test)
print(pred_lr)

lr_prob = lr.predict_proba(X_test)
print(lr_prob)

jaccard_score(y_test, pred_lr, pos_label=0)

def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix without normalization")
        print(cm)
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2
        for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j,i, format(cm[i, j], fmt), horizontalalignment = "center", color = "white" if cm[i,j] > thresh else "black")
            plt.tight_layout()
            plt.xlabel("Predicted label")
            plt.ylabel("Actual label")
            print(confusion_matrix(y_test, pred_lr, labels=[1,0]))

cnf_matrix = confusion_matrix(y_test, pred_lr, labels=[1,0])
np.set_printoptions(precision=2)
plt.figure()

plot_confusion_matrix(cnf_matrix, classes=["Donation in March 2007=1", "Donation in March 2007=0"], normalize=False, title="Confusion matrix data")
print(classification_report(y_test, pred_lr))
plt.show()
