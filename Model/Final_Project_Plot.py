#######################
from matplotlib.lines import _LineStyle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
#######################
#LogisticRegression
#######################
# Load the origin data 
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Fitting the Logistic Regression
LR = LogisticRegression(random_state=0)
LR.fit(X_train, y_train.values.ravel())
# Cross validation 
scores = cross_val_score(LR, X_train, y_train.values.ravel(),n_jobs = -1)
print(f" Accuracy of Logistic Regression : {round(np.mean(scores) ,4) * 100} %")
# Parameter optimization 
parameters = {'C':[0.001,0.01,0.1,1,10,100,1000],'solver':['newton-cg','lbfgs','sag','saga']}
solver = {'solver':['newton-cg','lbfgs','sag','saga']}
LR_grid = GridSearchCV(LR,parameters,scoring='accuracy',n_jobs = -1)
LR_grid.fit(X_train, y_train.values.ravel())
print(f"Best parameters of Logistic Regression : {LR_grid.best_params_}")
print(f"Best score of Logistic Regression : {LR_grid.best_score_}")
# Predict the test set
y_pred = LR_grid.predict(X_test)
# Save the y_pred to CSV
y_pred = pd.DataFrame(y_pred)
y_pred.to_csv('y_pred_LR.csv',index=False)
# Evaluate the model
print(f" Accuracy of Logistic Regression : {round(metrics.accuracy_score(y_test, y_pred) ,4) * 100} %")
print(f" Confusion Matrix of Logistic Regression : \n {metrics.confusion_matrix(y_test, y_pred)}")
print(f" Classification Report of Logistic Regression : \n {metrics.classification_report(y_test, y_pred)}")
# Plot the confusion matrix
sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix of Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()
# Plot the ROC curve
y_pred_prob = LR_grid.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of Logistic Regression')
plt.show()
# Save the ROC curve to CSV
fpr = pd.DataFrame(fpr)
fpr.to_csv('fpr_LR.csv',index=False)
tpr = pd.DataFrame(tpr)
tpr.to_csv('tpr_LR.csv',index=False)
# Save the PR as PR_LR
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_prob)
PR_LR = pd.DataFrame(np.array([precision,recall]).T)
PR_LR.to_csv('PR_LR.csv',index=False)


from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Set the path
path = 'C:/Users/LAN/Downloads/208notes/STA208_final_proj/Model'
# Load the data
y_pred_LR = pd.read_csv('y_pred_LR.csv')
y_pred_RF = pd.read_csv('y_pred_RF.csv')
y_pred_SVM = pd.read_csv('y_pred_SVM.csv')
ROC_RF = pd.read_csv('ROC_RF.csv')
ROC_svm = pd.read_csv('ROC_svm.csv')
fpr_LR = pd.read_csv('fpr_LR.csv')
tpr_LR = pd.read_csv('tpr_LR.csv')
PR_RF = pd.read_csv('PR_RF.csv')
PR_svm= pd.read_csv('PR_svm.csv')
PR_LR = pd.read_csv('PR_LR.csv')
# Plot Roc in one plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(ROC_RF.iloc[:,0], ROC_RF.iloc[:,1], label='Random Forest', color = 'blue')
ax.plot(ROC_svm.iloc[:,0], ROC_svm.iloc[:,1], label='SVM',color = 'orange')
ax.plot(fpr_LR, tpr_LR, label='Logistic Regression',color = 'green')
ax.legend(loc='lower right')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.arange(0, 1.1, 0.1))
# Zoom out and add annotation
axins = ax.inset_axes([0.2, 0.5, 0.3, 0.3])
axins.plot(ROC_RF.iloc[:,0], ROC_RF.iloc[:,1], label='Random Forest', color = 'blue')
axins.plot(ROC_svm.iloc[:,0], ROC_svm.iloc[:,1], label='SVM',color = 'orange')
axins.plot(fpr_LR, tpr_LR, label='Logistic Regression',color = 'green')
# Zoom in the upper left corner
axins.set_xlim(0, 0.2)
axins.set_ylim(0.8, 1)
axins.set_xticks([])
axins.set_yticks([])
# Add connnecting lines
mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="k")
plt.show()
fig.savefig('ROC.png', dpi=500)


# Plot Precision-Recall in one plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(PR_RF.iloc[:,0], PR_RF.iloc[:,1], label='Random Forest', color = 'blue')
ax.plot(PR_svm.iloc[:,0], PR_svm.iloc[:,1], label='SVM',color = 'orange')
ax.plot(PR_LR.iloc[:,0], PR_LR.iloc[:,1], label='Logistic Regression',color = 'green')
ax.legend(loc='lower right')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.set_xticks(np.arange(0.5, 1.05, 0.1))
ax.set_yticks(np.arange(0, 1.1, 0.1))
# Zoom out and add annotation
axins = ax.inset_axes([0.5, 0.5, 0.3, 0.3])
axins.plot(PR_RF.iloc[:,0], PR_RF.iloc[:,1], label='Random Forest')
axins.plot(PR_svm.iloc[:,0], PR_svm.iloc[:,1], label='SVM')
axins.plot(PR_LR.iloc[:,0], PR_LR.iloc[:,1], label='Logistic Regression')
# Zoom in the upper right corner
axins.set_xlim(0.9, 1)
axins.set_ylim(0.8, 1)
axins.set_xticks([])
axins.set_yticks([])
# Add connnecting lines
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="k")
plt.show()

# Save the ROC and PR plot

fig.savefig('PR.png', dpi=500)

