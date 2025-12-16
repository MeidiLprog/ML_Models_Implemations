import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.decomposition import PCA

da = load_iris()

dt = pd.DataFrame(data=da.data, columns=da.feature_names)
dt["target"] = da.target
mean_by_class = dt.groupby("target")["sepal length (cm)"].mean()
second = pd.DataFrame({

    "Names" : ["john","mickael","meidi","melissa"],
    "Age" : [20,20,25,22]

})

print(f"Number of null {dt.isna().sum()}\n")

plt.figure(figsize=(10,5))
print(dt.head())
print(dt.describe())

#bar plot
name = ["setosa","versicolor","virginica"]
plt.bar(name,mean_by_class.values)
plt.xlabel("Observations")
plt.ylabel("Sepal Length")
plt.title("Barplot of special length")
plt.show()

print()
X = da.data
pc = PCA(n_components=min(X.shape[0],X.shape[1]))
pc.fit(X) #search the factorial axes

ratio = pc.explained_variance_ratio_
print(ratio)
somme = 0
tab = []
[tab.append(somme := somme + i) for i in ratio if somme <= 0.95]
print(tab)
new_t = (tab[:2] if len(tab) < 3 else tab[:3])
print(new_t)

pc = PCA(n_components=2)
pc.fit(X)

projection_pca = pc.transform(X) #projects our individuals

plt.scatter(projection_pca[:,0],projection_pca[:,1],c=da.target)
plt.xlabel("Axe 1")
plt.ylabel("Axe 2")
plt.show()


#logistic regression


model = LogisticRegression(max_iter=200)

y = dt["target"]
X = dt[da.feature_names]

X_train, X_test, y_train,y_test = train_test_split(X,y, random_state=42,test_size=0.20)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test,y_pred)} \n")
print(f"Precision : {precision_score(y_test,y_pred, average='macro')} \n")
print(f"Recall : {recall_score(y_test,y_pred, average='macro')} \n")

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)
plt.title("Confusion Matrix")
plt.show()


#let's build a ROC

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

mask = (y != 2)
X_bin = X[mask]
Y_bin = y[mask]

Y_bin = (Y_bin == 1).astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000)

y_scores = cross_val_predict(
    model,
    X_bin,
    Y_bin,
    cv=skf,
    method="predict_proba"
)[:, 1]

fpr, tpr, thresholds = roc_curve(Y_bin, y_scores)
auc = roc_auc_score(Y_bin, y_scores)
plt.figure(figsize=(10,6))
plt.xlabel("FPR")
plt.ylabel("TPR/RECALL")
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label="Random baseline")
plt.show()
print("AUC :", auc)

