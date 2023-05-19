import pandas as pd
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
from sklearn.metrics import f1_score
import pickle
import os.path
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN
from os import path
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
from sklearn.feature_selection import SelectPercentile, chi2, SelectKBest, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from tkinter import *
from PIL import ImageTk,Image
import warnings
top = Tk()
top.title("Diabtes")
top.iconbitmap()
top.minsize(1550,900)

warnings.filterwarnings('ignore')

res={};
for i in range(0,40):
    res[i]=Label(bg="#CB24FF");

def save_object(obj, filename):
    pickle.dump(obj, open(filename, 'wb'))


def load_object(filename):
    return pickle.load(open(filename, 'rb'))

def delete():
    for i in range(0,40):
        res[i].pack_forget()


def LG_func():
    delete()
    print("------------Logistic Regression------------")
    if str(path.isfile('LG_Model.sav')) == False:
        model = LogisticRegression()
        model.fit(X_train, Y_train)
        save_object(model, "LG_Model.sav")
    else:
        model = load_object("LG_Model.sav")
    Y_pred = model.predict(X_test)
    res[0]=Label(text="Accuracy Score:"+str(accuracy_score(Y_test, Y_pred)))
    # confusion matrix
    confusion_mat = confusion_matrix(Y_test, Y_pred)
    res[1] = Label(text="Confusion matrix"+str(confusion_mat))
    res[2] = Label(text= str(classification_report(Y_test, Y_pred)))
    res[3] = Label(text="F1-Score: "+ str(f1_score(Y_test, Y_pred)))
    print("Accuracy Score:",accuracy_score(Y_test,Y_pred))
    #confusion matrix
    confusion_mat = confusion_matrix(Y_test,Y_pred)
    print("Confusion matrix",confusion_mat)
    print(classification_report(Y_test, Y_pred))
    print("F1-Score: ",f1_score(Y_test, Y_pred))
    print("Score: ",model.score(X_test,Y_test))
    pyplot.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=True, cmap="YlGnBu")
    pyplot.show()

def LG_func_show():
    delete()
    for i in range(0,4):
        res[i].place(x=10,y=150)

def SVM_func():
    delete()
    print("-------------SVM-------------------------")
    if str(path.isfile('SVM_Model.sav')) == False:
        SVM = svm.LinearSVC()
        SVM.fit(X_train, Y_train)
        save_object(SVM, "SVM_Model.sav")
    else:
        SVM = load_object("SVM_Model.sav")
    predictscm = SVM.predict(X_test)
    res[4] = Label(text = "Accuracy Score:"+ str(accuracy_score(Y_test, predictscm)))
    res[5] = Label(text = "precision"+ str(predictscm))
    res[6] = Label(text="Accuracy Score:"+str (accuracy_score(Y_test, predictscm)))
    res[7] = Label(text= "precision"+str(predictscm))
    from sklearn import metrics
    res[8] = Label(text= str(classification_report(Y_test, predictscm)))
    res[9] = Label(text = "F1-score: "+ str(metrics.f1_score(Y_test, predictscm)))
    print("Accuracy Score:", accuracy_score(Y_test, predictscm))
    print("precision",predictscm)
    print("Accuracy Score:", accuracy_score(Y_test, predictscm))
    print("precision",predictscm)
    from sklearn import metrics
    print(classification_report(Y_test, predictscm))
    print("F1-score: ",metrics.f1_score(Y_test,predictscm))
    print("recall", metrics.recall_score(Y_test, predictscm))
    pyplot.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(Y_test, predictscm), annot=True, cmap="YlGnBu")
    pyplot.show()

def SVM_func_show():
    delete()
    for i in range(4,10):
        res[i].place(x=350,y=150)



def knn_func():
    delete()
    print("---------------KNN---------------------")
    if str(path.isfile('KNN_Model.sav')) == False:
        knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
        knn.fit(X_train, Y_train)
        save_object(knn, "KNN_Model.sav")
    else:
        knn = load_object("KNN_Model.sav")
    y_predK = knn.predict(X_test)
    res[10] = Label(text= str(confusion_matrix(Y_test, y_predK)))
    res[11] = Label(text= "Accuracy Score:"+ str(accuracy_score(Y_test, y_predK)))
    res[12] = Label(text= str(classification_report(Y_test, y_predK)))
    res[13] = Label(text= "F1-score: "+str(metrics.f1_score(Y_test, y_predK)))
    print(confusion_matrix(Y_test, y_predK))
    print("Accuracy Score:", accuracy_score(Y_test, y_predK))
    print(classification_report(Y_test, y_predK))
    print("F1-score: ",metrics.f1_score(Y_test,y_predK))
    pyplot.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(Y_test, y_predK), annot=True, cmap="YlGnBu")
    pyplot.show()

def knn_func_show():
    delete()
    for i in range(10,14):
        res[i].place(x=700,y=150)


def scalling_func():
    delete()
    print("---------------Data Scalling---------------------")
    ss = StandardScaler()
    scaled = ss.fit_transform(X_test)
    print(scaled)
    res[14] = Label(text=str(scaled))
    dataset = DataFrame(scaled)
    res[15] = Label(text=str(dataset.describe()))
    print(dataset.describe())
    # histograms of the variables
    dataset.hist()
    pyplot.show()

def scalling_func_show():
    delete()
    for i in range(14,16):
        res[i].place(x=200,y=550)


def Decison_func():
    delete()
    print("-------------------Decison tree---------------")
    from sklearn.tree import DecisionTreeClassifier
    Dtree = load_object("DModel.sav")
    predicition = Dtree.predict(X_test)
    res[16] = Label(text= str(predicition))
    from sklearn import metrics
    res[17] = Label(text="Accuracy Score:"+ str(accuracy_score(Y_test, predicition)))
    from sklearn.metrics import classification_report, confusion_matrix
    res[18] = Label(text= str(confusion_matrix(Y_test, predicition)))
    res[19] = Label(text= str(classification_report(Y_test, predicition)))
    res[20] = Label(text= "F1-Score: "+ str(f1_score(Y_test, predicition)))
    confusion_mat = confusion_matrix(Y_test,predicition)
    print("Confusion matrix",confusion_mat)
    print(classification_report(Y_test, predicition))
    print("F1-Score: ",f1_score(Y_test, predicition))
    print("Score: ",Dtree.score(X_test,Y_test))
    pyplot.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(Y_test, predicition), annot=True, cmap="YlGnBu")
    pyplot.show()

def Decison_func_show():
    delete()
    for i in range(16,21):
        res[i].place(x=1000,y=150)



def testing():
    dtestx=pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
    print(dtestx.shape)
    dtestx = dtestx.dropna()
    print(dtestx.shape)
    dtestx=dtestx.drop_duplicates()
    print(dtestx.shape)
    print(dataset.describe())
    x2=dtestx.iloc[:,1:21]
    y2=dtestx['Diabetes_binary']
    lsvc = svm.LinearSVC(C=0.0001, penalty="l1", dual=False).fit(x2, y2)
    FeatureSelection = SelectFromModel(lsvc, prefit=True)
    x2 = FeatureSelection.transform(x2)
    ss = StandardScaler()
    scaled = ss.fit_transform(x2)
    x2=scaled
    print(scaled)
    dataset = DataFrame(scaled)
    model = load_object("LG_Model.sav")
    Dtree = load_object("DModel.sav")
    SVM = load_object("SVM_Model.sav")
    knn = load_object("KNN_Model.sav")
    print("-------------------Logistic Regression---------------")
    lgscore=model.score(x2,y2)
    lgpredict=model.predict(x2)
    print (classification_report(y2, lgpredict))
    res[21] = Label(text= "Accuracy Score:"+str(lgscore))
    res[22] = Label(text= "classification_report: "+str(classification_report(y2, lgpredict)))
    print(lgscore)
    print("-------------------Decison tree---------------")
    Dscore=Dtree.score(x2,y2)
    Dpredict=Dtree.predict(x2)
    print ("classification_report: ",classification_report(y2, Dpredict))
    print("Score: ",Dscore)
    res[23] = Label(text= "Accuracy Score:"+str(Dscore))
    res[24] = Label(text= "classification_report: "+str(classification_report(y2, Dpredict)))
    print("-------------------SVM---------------")
    Svmscore=SVM.score(x2,y2)
    Svmpredict=SVM.predict(x2)
    print ("classification_report: ",classification_report(y2, Svmpredict))
    print("Score: ",Svmscore)
    res[25] = Label(text= "Accuracy Score:"+str(Svmscore))
    res[26] = Label(text= "classification_report: "+str(classification_report(y2, Svmpredict)))
    print("-------------------KNN---------------")
    Kscore=knn.score(x2,y2)
    Kpredict=SVM.predict(x2)
    print ("classification_report: ",classification_report(y2, Kpredict))
    print("Score: ",Kscore)
    res[27] = Label(text= "Accuracy Score:"+str(Kscore))
    res[28] = Label(text= "classification_report: "+str(classification_report(y2, Kpredict)))


def Test_func_show():
    delete()
    for i in range(21,29):
        res[i].place(x=900,y=550)

diabetes_clean = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv', sep=',',header=0)
dataset2 = diabetes_clean.iloc[:, :-1]
print("# of Rows, # of Columns: ", dataset2.shape)
print("\nColumn Name           # of Null Values\n")
print(dataset2.isnull().sum())
print((dataset2[:] == 0).sum())
# clean data
df = diabetes_clean.drop_duplicates()
df['Diabetes_binary'] = df['Diabetes_binary'].fillna(df['Diabetes_binary'].mean())
df['HighBP'] = df['HighBP'].fillna(df['HighBP'].mean())
df['HighChol'] = df['HighChol'].fillna(df['HighChol'].mean())
df['CholCheck'] = df['CholCheck'].fillna(df['CholCheck'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].median())
df['Smoker'] = df['Smoker'].fillna(df['Smoker'].mean())
df['Stroke'] = df['Stroke'].fillna(df['Stroke'].mean())
print(df.isnull().sum())
df = diabetes_clean.drop_duplicates()
print((df.corr()))
if str(path.isfile('X-Featurebalancing.sav')) == False:
    x = df.iloc[:, 1:21]
    y = df.iloc[:, 0]
    lsvc = svm.LinearSVC(C=0.0001, penalty="l1", dual=False).fit(x, y)
    FeatureSelection = SelectFromModel(lsvc, prefit=True)
    x_new = FeatureSelection.transform(x)
    print(FeatureSelection.get_support())
    sample = SMOTEENN(sampling_strategy=0.5)
    # fit and apply the transform
    X_over, y_over = sample.fit_resample(x_new, y)
    save_object(X_over, "X-Featurebalancing.sav")
    save_object(y_over, "Y-Featurebalancing.sav")
else:
    print("-----------------loaded-------------------")
    X_over = load_object("X-Featurebalancing.sav")
    y_over = load_object("Y-Featurebalancing.sav")

print("succes")
# GUI



X_train, X_test, Y_train, Y_test = train_test_split(X_over, y_over, test_size=0.30, random_state=42)
LG_func()
but_logistic = Button(text="Logistic Regression", command=LG_func_show)
but_logistic.place(x=10,y=100)

Decison_func()
but_Decison = Button(text="Decison tree", command=Decison_func_show)
but_Decison.place(x=1000,y=100)
SVM_func()
but_svm = Button(text="SVM", command=SVM_func_show)
but_svm.place(x=350,y=100)
# Kneighbors
knn_func()
but_knn = Button(text="KNN", command=knn_func_show)
but_knn.place(x=700,y=100)
# Data Scalling

scalling_func()
but_scaling = Button(text="Data Scalling", command=scalling_func_show)
but_scaling.place(x=200,y=500)

testing()
but_Test = Button(text="Testing New Model", command=Test_func_show)
but_Test.place(x=900,y=500)

top.mainloop()

#predict input
#Diabetes_binary= input("Diabetes_binary: ")
#HighBP = input("HighBP: ")
#CholCheck = input("CholCheck: ")
#HeartDiseaseorAttack = input("HeartDiseaseorAttack: ")
#MentHlth = input("MentHlth: ")
#PhysHlth = input("PhysHlth: ")
#DiffWalk = input("DiffWalk: ")
#Education = input("Education: ")
#Income = input("Income: ")
#first_input=[Diabetes_binary, HighBP, CholCheck, HeartDiseaseorAttack, MentHlth,PhysHlth,DiffWalk,Education,Income]
#new_input = [first_input]
#yhat = model.predict(new_input)
#print(yhat[0])