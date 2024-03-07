from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, f1_score, auc, classification_report , make_scorer
from sklearn.model_selection import cross_val_score

from scipy.ndimage import gaussian_filter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from datas import *

import warnings
warnings.filterwarnings("ignore")


#####################################################################################


def create_train_test_data(df_preprocessed, type_vectorizer=TfidfVectorizer, binary=False, ngrams=(1,2)):
    """Effectue un train test split des données

    Args:
        df_preprocessed (dataframe): dataframe des documents prétraités
        type_vectorizer (sklearn.vectorizer, optional): le type de vectorizer
        binary (bool, optional): Si on doit utiliser un BoW binaire ou non (possible uniquement avec CountVectorizer)
        ngrams (tuple, optional): la portée des n-grams

    Returns:
        Bags of Words des exemples et les labels, pour tout le dataset  et les ensembles de train et test
    """
    if(not binary):
        vectorizer = type_vectorizer(ngram_range=ngrams)
    else : 
        vectorizer = type_vectorizer(ngram_range=ngrams,binary=True)
    X = vectorizer.fit_transform(df_preprocessed.text)
    Y = df_preprocessed.iloc[:,-1]

    rs=10
    [X_train, X_test, y_train, y_test]  = train_test_split(X, Y, test_size=0.2, random_state=rs, shuffle=True)

    return X, Y, X_train, X_test, y_train, y_test 

def f1_score_class_0(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None)[0]


#####################################################################################


def tester_nb(X, Y, X_train, y_train, X_test, y_test, metric="classification_report"):
    """Entraîne et teste un modèle naive bayes, et retourne le score de la métrique donnée

    Args:
        X (scipy sparse matrix): Bag of Words de tout le dataset
        Y (liste): liste de tous les labels du dataset 
        X_train (scipy sparse matrix): BoW de la partie train
        y_train (liste): liste des labels de la partie train
        X_test (scipy sparse matrix): BoW de la partie test
        y_test (liste): liste des labels de la partie test
        metric (str, optional): métrique d'évaluation. Defaults to "classification_report".

    Returns:
        float ou None: Score de la métrique demandée, ou imprime le rapport de classification 
    """
    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    pred_nb = nb_clf.predict(X_test)

    
    resultat_metrique = None
    if metric=="classification_report":
        print(classification_report(y_test, pred_nb, target_names=['Classe négative', 'Classe positive']))
    elif metric=="accuracy":
        acc=accuracy_score(y_test, pred_nb)
        print("Accuracy score = ",acc)
        resultat_metrique=acc
    elif metric=="f1":
        f1=f1_score(y_test, pred_nb, average=None) # average=None et f1[0] pour selectionner la classe de mitterand
        print("F1 score classe négative = ",f1[0])
        print("F1 score classe positive = ",f1[1])
        resultat_metrique=f1[0]
    elif metric=="cross_val_score_movies":
        scores = cross_val_score( nb_clf, X, Y, cv=5)
        print("score cross validation: ", scores.mean())
        resultat_metrique=scores.mean()
    elif metric=="cross_val_score_president":
        scorer = make_scorer(f1_score_class_0)
        scores = cross_val_score( nb_clf, X, Y, cv=5, scoring=scorer)
        print("score cross validation: ", scores.mean())
        resultat_metrique=scores.mean()
    print()
    return resultat_metrique

def tester_lr(X, Y, X_train, y_train, X_test, y_test, class_weight=None, nmax_iter=1000, tracer_courbes = False, metric="classification_report"): 
    """Entraîne et teste un modèle de régression logistique, et retourne le score de la métrique donnée

    Args:
        X (scipy sparse matrix): Bag of Words de tout le dataset
        Y (liste): liste de tous les labels du dataset 
        X_train (scipy sparse matrix): BoW de la partie train
        y_train (liste): liste des labels de la partie train
        X_test (scipy sparse matrix): BoW de la partie test
        y_test (liste): liste des labels de la partie test
        nmax_iter (int): nombre d'itérations max
        metric (str, optional): métrique d'évaluation.

    Returns:
        float ou None: Score de la métrique demandée, ou imprime le rapport de classification 
    """
    
    t = 1e-8
    C=100.0
    lr_clf = LogisticRegression(random_state=0, solver='liblinear', class_weight=class_weight , max_iter=nmax_iter, tol=t, C=C)
    lr_clf.fit(X_train, y_train)
    pred_lr = lr_clf.predict(X_test)

    if tracer_courbes == True:
        y_probs = lr_clf.predict_proba(X_test)[:, 1]
        tracer_courbe_roc(y_test, y_probs)
        tracer_courbe_rappel_precision(y_test, y_probs)
    
    resultat_metrique = None
    if(metric=="classification_report"):
        print(classification_report(y_test, pred_lr, target_names=['Classe négative', 'Classe positive']))
    elif(metric=="accuracy"):
        acc=accuracy_score(y_test, pred_lr)
        print("Accuracy score = ",acc)
        resultat_metrique=acc
    elif(metric=="f1"):
        f1=f1_score(y_test, pred_lr, average=None) # average=None et f1[0] pour selectionner la classe de mitterand
        print("F1 score classe négative = ",f1[0])
        print("F1 score classe positive = ",f1[1])
        resultat_metrique=f1[0]
    elif metric=="cross_val_score_movies":
        scores = cross_val_score( lr_clf, X, Y, cv=5)
        print("score cross validation: ", scores.mean())
        resultat_metrique=scores.mean()
    elif metric=="cross_val_score_president":
        scorer = make_scorer(f1_score_class_0)
        scores = cross_val_score( lr_clf, X, Y, cv=5, scoring=scorer)
        print("score cross validation: ", scores.mean())
        resultat_metrique=scores.mean()

    print()
    return resultat_metrique

def tester_svm(X, Y, X_train, y_train, X_test, y_test, class_weight=None, nmax_iter=1000, tracer_courbes=False, metric="classification_report"):    
    """Entraîne et teste un modèle de SVC linéaire, et retourne le score de la métrique donnée

    Args:
        X (scipy sparse matrix): Bag of Words de tout le dataset
        Y (liste): liste de tous les labels du dataset 
        X_train (scipy sparse matrix): BoW de la partie train
        y_train (liste): liste des labels de la partie train
        X_test (scipy sparse matrix): BoW de la partie test
        y_test (liste): liste des labels de la partie test
        nmax_iter (int): nombre d'itérations max
        metric (str, optional): métrique d'évaluation.

    Returns:
        float ou None: Score de la métrique demandée, ou imprime le rapport de classification 
    """
    
    svm_clf = LinearSVC(random_state=0, class_weight=class_weight, max_iter=nmax_iter)
    svm_clf.fit(X_train, y_train)
    pred_svm = svm_clf.predict(X_test)

    resultat_metrique = None
    if(metric=="classification_report"):
        print(classification_report(y_test, pred_svm, target_names=['Classe négative', 'Classe positive']))
    elif(metric=="accuracy"):
        acc=accuracy_score(y_test, pred_svm)
        print("Accuracy score = ",acc)
        resultat_metrique=acc
    elif(metric=="f1"):
        f1=f1_score(y_test, pred_svm, average=None) # average=None et f1[0] pour selectionner la classe de mitterand
        print("F1 score classe négative = ",f1[0])
        print("F1 score classe positive = ",f1[1])
        resultat_metrique=f1[0]
    elif metric=="cross_val_score_movies":
        scores = cross_val_score( svm_clf, X, Y, cv=5)
        print("score cross validation: ", scores.mean())
        resultat_metrique=scores.mean()
    elif metric=="cross_val_score_president":
        scorer = make_scorer(f1_score_class_0)
        scores = cross_val_score( svm_clf, X, Y, cv=5, scoring=scorer)
        print("score cross validation: ", scores.mean())
        resultat_metrique=scores.mean()

    if tracer_courbes == True:
        y_scores = svm_clf.decision_function(X_test)
        tracer_courbe_roc(y_test, y_scores)
        tracer_courbe_rappel_precision(y_test, y_scores)
    return resultat_metrique


#####################################################################################


def tester_models(X, Y, X_train, y_train, X_test, y_test, nmax_iter=1000, metric="classification_report"):
    """Execute la fonction de test de chaque modèle

    Args:
        X (scipy sparse matrix): Bag of Words de tout le dataset
        Y (liste): liste de tous les labels du dataset 
        X_train (scipy sparse matrix): BoW de la partie train
        y_train (liste): liste des labels de la partie train
        X_test (scipy sparse matrix): BoW de la partie test
        y_test (liste): liste des labels de la partie test
        nmax_iter (int, optional): nombre d'itérations max
        metric (str, optional): métrique d'évaluation.

    Returns:
        tuple (float ou None) : Résultat du test de chaque modèle
    """
    
    print('---- Naive Bayes ----')
    resultat_metrique_nb=tester_nb(X, Y, X_train, y_train, X_test, y_test, metric=metric)
    
    print('---- Logistic regression ---- ')
    resultat_metrique_lr=tester_lr(X, Y, X_train, y_train, X_test, y_test, nmax_iter=nmax_iter, metric=metric)
    
    print('---- SVC ---- ')
    resultat_metrique_svc=tester_svm(X, Y, X_train, y_train, X_test, y_test, nmax_iter=nmax_iter,metric=metric)
    
    return resultat_metrique_nb, resultat_metrique_lr, resultat_metrique_svc

def tester(df_preprocessed, etape=1, *args):
    """Divise le dataset en train et test et test effectue une étape de test

    Args:
        df_preprocessed (dataframe): dataframe des documents prétraités
        etape (int, optional): l'étape de sélection des paramètres et modèles.
            etape 1: Tester tous les modèles et toutes les variantes bag of words 
                     (On l'effectue deux fois, selon le niveau de suppression des stopwords)
            etape 2: Tester tous les modèles sur la variante BoW pour déterminer les 
                     meilleurs paramètres de la variante et le meilleur modèle de machine learning
                     
    Returns:
        data: Dataframe des résultats des tests
    """
    
    if(etape==1):
        #BOW
        print("******************** BOW ********************")
        X, Y, X_train, X_test, y_train, y_test = create_train_test_data(df_preprocessed, type_vectorizer=CountVectorizer)
        tester_models(X, Y, X_train, y_train, X_test, y_test, nmax_iter=1000)

        #BOW binaire
        print("******************** BOW binaire ********************")
        X, Y, X_train, X_test, y_train, y_test =create_train_test_data(df_preprocessed, type_vectorizer=CountVectorizer, binary=True)
        tester_models(X, Y, X_train, y_train, X_test, y_test, nmax_iter=1000)

        #TF-IDF
        print("******************** TF-IDF ********************")
        X, Y, X_train, X_test, y_train, y_test =create_train_test_data(df_preprocessed, type_vectorizer=TfidfVectorizer)
        tester_models(X, Y, X_train, y_train, X_test, y_test, nmax_iter=1000)
    
    elif(etape==2):
        # *args doit etre sous la forme [dataset (movies ou president), metric (accuracy, cross_val_score_nomDataset, classification_report)]
        dataset = args[0]
        metric=args[1]

        if(dataset=="movies"):
            params = { 'list_max_df':[0.7, 0.75, 0.8, 0.85],
                    'list_min_df':[2,3,4,5,6],
                    'ngram_range':[(1,2),(1,3)]

            }
        else :
            params = { 'list_max_df':[0.6,0.7, 0.75],
                    'list_min_df':[2,5,7,10],
                    'ngram_range':[(1,2),(1,3),(1,4)]
            }
        data = {
            'max_df': [],
            'min_df': [],
            'ngram_range': [],
            'Naive Bayes': [],
            'Logistic Regression': [],
            'SVC': []
        }
        for max_df in params.get('list_max_df'):
            for min_df in params.get("list_min_df"):
                for ngram in params.get('ngram_range'):
                    print()
                    print("max_df",max_df, "min_df=",min_df, "ngram_range=",ngram)
                    print("________________________________________________________")
                    if(dataset=="movies"):
                        vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram, binary=True)
                    else:
                        vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, ngram_range=ngram, binary=False)
                        
                    X = vectorizer.fit_transform(df_preprocessed.text)
                    Y = df_preprocessed.iloc[:,-1]
                    
                    rs=10
                    [X_train, X_test, y_train, y_test]  = train_test_split(X, Y, test_size=0.2, random_state=rs, shuffle=True)
                    
                    resultat_metrique_nb, resultat_metrique_lr, resultat_metrique_svc=tester_models(X, Y, X_train, y_train, X_test, y_test, nmax_iter=1000, metric=metric)
                    data['max_df'].append(max_df)
                    data['min_df'].append(min_df)
                    data['ngram_range'].append(ngram)
                    data["Naive Bayes"].append(resultat_metrique_nb)
                    data['Logistic Regression'].append(resultat_metrique_lr)
                    data['SVC'].append(resultat_metrique_svc)
        return pd.DataFrame(data)


#####################################################################################


def tracer_courbe_roc(y_test, y_probs):
    
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)

    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def tracer_courbe_rappel_precision(y_test, y_probs):

    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve (AUC = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


#####################################################################################


def gaussian_smoothing(pred, size):
    """Applique un lissage gaussien sur les probabilités

    Args:
        pred (ndarray): Prédictions de la classe
        size (float): taille du filtre gaussien

    Returns:
        ndarray: prédictions lissées
    """
    smoothed_pred = gaussian_filter(pred, sigma=size)
    return smoothed_pred
