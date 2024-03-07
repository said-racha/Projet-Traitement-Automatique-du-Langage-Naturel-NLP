from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import numpy as np

import warnings
warnings.filterwarnings("ignore")

def get_vocabulaire(df, type_vectorizer=CountVectorizer,  top=None, min__df=1, max__df=0.99, n_gramme=(1,1)):    
    """Crée une représentation Bag of Words des documents et retourne des informations sur le vocabulaire

    Args:
        df (pd.DataFrame): dataframe contenant les documents (phrases) associés à leurs labels
        type_vectorizer (sklearn vectorizer, optional): le type du vectorizer : CountVectorizer ou TfidfVectorizer
        top (int, optional): considérer uniquement les top mots ayant le meilleur score
        min__df (int, optional): quantité minimum d'apparition documentaire
        max__df (float, optional): quantité maximum d'apparition documentaire
        n_gramme (tuple, optional): la portée des n-grammes à considérer

    Returns:
        dic_size: taille du vocabulaire
        mots: liste des mots du vocabulaire
        dictionnaire_mots_freq: mots du vocabulaire associés à leurs fréquences respectives
    """
    vectorizer = type_vectorizer(max_features=top, ngram_range=n_gramme, min_df=min__df, max_df=max__df)
    X = vectorizer.fit_transform(df.text)  
    dic_size = X.shape[1]
    mots = vectorizer.get_feature_names_out()
    dictionnaire_mots_freq = dict(zip(mots, X.sum(axis=0).tolist()[0]))
    return dic_size, mots, dictionnaire_mots_freq

###########################

def visualiser_wordcloud(dictionnaire_mots):
    """visualisation wordcloud

    Args:
        dictionnaire_mots (dictionnaire {mot:frequence}): mots associés à leur fréquence
    """
    nuage_mots_1 = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dictionnaire_mots)

    plt.figure(figsize=(10, 5))
    plt.imshow(nuage_mots_1, interpolation='bilinear')
    plt.axis('off')
    plt.show()

############################

def get_mots_discriminants_odds_ratio(classe_interet, nom_classe, df_preprocessed, mots_preprocessed, dic_size_classe_interet, dic_size_autre, N_mots_discriminants=100 ):
    """Retourne les mots qui discriminent le plus une classe au sens des odds ratio

    Args:
        classe_interet (int): La classe 'positive' pour le odds ratio (celle qu'on veut mettre en évidence). Présidents: -1, 1; Movies: 0, 1
        nom_classe (str): nom de la classe ('polarite' ou 'president')
        df_preprocessed (dataframe): dataframe des documents prétraités
        mots_preprocessed (numpy.ndarray): Les mots du vocabulaire prétraité
        dic_size_classe_interet (int): Taille du vocabulaire de la classe d'intérêt
        dic_size_autre (int): Taille du dictionnaire de la classe opposée
        N_mots_discriminants (int, optional): Nombre de mots à sélectionner

    Returns:
        mots_plus_discriminants : Liste des mots les plus discriminants pour la classe demandée
        dict_odds_ratio_final : Dictionnaire des mots avec leur score odds ratio pour la classe demandée
    """
    
    vectorizer = CountVectorizer(min_df=10, max_df=0.9)
    X = vectorizer.fit_transform(df_preprocessed.text)
    
    # Fréquences de chaque mot pour chaque classe
    freq_mots_classe0 = np.array(X.toarray()[np.array(df_preprocessed[nom_classe]) != classe_interet].sum(axis=0))
    freq_mots_classe1 = np.array(X.toarray()[np.array(df_preprocessed[nom_classe]) == classe_interet].sum(axis=0))
    
    # Calculer l'odds ratio pour chaque mot   (Plus il est élevé plus le mot discrimine la classe 1)
    p = freq_mots_classe1 / dic_size_classe_interet     #Note : dic_size_classe_interet est la taille du vocabulaire de l'interlocteur dont on cherche à trouver les mots discriminants
    q = freq_mots_classe0 / dic_size_autre         
    
    # Pour éviter les divisions par zéro
    if type(q) != int: 
        if q.all() == 0: q += 1e-5 #np.array([1e-5]*len(q))  #q.all()=1e-5
    else :
        if q == 0: q = 1e-5
        
    odds_ratio = (p/(1-p)) / (q/(1-q))

    dictionnaire_odds_ratio = dict(zip(mots_preprocessed, odds_ratio))

    dictionnaire_odds_ratio_trie = dict(sorted(dictionnaire_odds_ratio.items(), key=lambda item: item[1], reverse=True))
    
    # Transformer les inf en un nombre (max*2)
    max_odd_ratio = 0
    for k,v in dictionnaire_odds_ratio_trie.items():
        if (v!=float('inf')): 
            max_odd_ratio=v
            break
            
    dict_odds_ratio_final=dictionnaire_odds_ratio_trie.copy()
    for k,v in dictionnaire_odds_ratio_trie.items():
        if (v==float('inf')):
            dict_odds_ratio_final[k] = max_odd_ratio*2

    mots_plus_discriminants = list(dictionnaire_odds_ratio_trie.keys())[:N_mots_discriminants]

    return mots_plus_discriminants, dict_odds_ratio_final

############################

def tracer_loi_de_zipf(rangs, y_positif, y_negatif, label_positif='Chirac', label_negatif='Mitterand'):
    fig, axis = plt.subplots(1, 2, figsize=(10, 5))

    axis[0].plot(rangs, y_positif, label=label_positif)
    axis[0].set_xlabel('Rang')
    axis[0].set_ylabel('Fréquence')
    axis[0].plot(rangs, y_negatif, label=label_negatif)
    axis[0].set_title(f'Loi de Zipf pour et {label_negatif}')
    axis[0].legend()

    axis[1].plot(np.log(rangs), np.log(y_positif), label=label_positif)
    axis[1].set_xlabel('log(Rang)')
    axis[1].set_ylabel('log(Fréquence)')
    axis[1].plot(np.log(rangs), np.log(y_negatif), label=label_negatif)
    axis[1].set_title(f'Log de la loi de Zipf pour {label_positif} et {label_negatif}')
    axis[1].legend()

    plt.tight_layout()
    plt.show()

