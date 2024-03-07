import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


##############################################################


## Nettoyage  des caracteres non alphabetiques 
def nettoyage(texte):
    #supprimer les caracteres speciaux et la ponctuation
    texte_propre = re.sub(r'[^a-z\s]', '', texte)
    return texte_propre


def read_stopwords(chemin_fichier):

    stopwords = []

    # Ouverture du fichier en mode lecture
    with open(chemin_fichier, 'r') as fichier:
        # Lecture du fichier ligne par ligne
        for ligne in fichier:
            # Nettoyage de la ligne (suppression des espaces en début et fin, et des sauts de ligne)
            mot = ligne.strip()
            # Ajout du mot à la liste des stopwords
            stopwords.append(mot)
    
    return set(stopwords)
 

###############################

stopwords_french =  read_stopwords("stopwords-fr-iso.txt")   #~ 700 stopwords
stopwords_french = stopwords_french.union(set(['savoir', 'devoir', 'falloir', 'faire'])) 
caracteres_fichier = set(['\n', '\t', '\r\n'])


############################preprocessing Président##########################


def preprocess_president(df_input, spacy_model_size='sm', stop_words=True):
    """Prétraitement du dataset Président. lemmatisation, tokenisation, stopwords, minuscules, suppression des caractères non alphabétiques

    Args:
        df_input (dataframe): dataframe des documents originaux
        spacy_model_size (str): charger le petit ou le grand modèle de langue de spacy
        stop_words (bool): niveau de suppression des stopwords
        
    Returns:
        dataframe: dataframe prétraité
    """
    
    """
    Note : 
    - Nous avons choisi SpaCy pour la lemmatization car il est plus adapté que nltk :
    "j'ai" reste la même avec nltk, mais avec spacy elle devient "je avoir", 
    et "qu'il" devient "que il", c'est plus pratique pour maintenant effectuer l'élimination des stopwords
    """
    
    df = df_input.copy()
    
    if spacy_model_size == 'lg':
        nlp = spacy.load("fr_core_news_lg")  
    elif spacy_model_size == 'sm':
        nlp = spacy.load("fr_core_news_sm")

    
    ## Tokenization et lemmatisation et  Suppression des stop words
    
    def tokenization_lemmatization_stopWords(texte):
        doc = nlp(texte)
        
        stopwords_fr_nltk = set(stopwords.words('french'))
        tok_lemm_text = []
        
        for token in doc:
            mot_lemmatised = token.lemma_
            
            if stop_words:   #Si stop_words == True, on utilise le gros fichier, sinon on utilise la petite liste de nltk
                if mot_lemmatised not in stopwords_french.union(caracteres_fichier):
                    tok_lemm_text.append(mot_lemmatised)
            elif mot_lemmatised not in stopwords_fr_nltk.union(caracteres_fichier):
                tok_lemm_text.append(mot_lemmatised)
                      
                

        return " ".join(tok_lemm_text)
    
    df.text= df.text.apply(lambda text:tokenization_lemmatization_stopWords(text))
    
    ## Transformer les accents en lettres non accentuées  
    df.text=df.text.apply(lambda x:x.lower())  
    df.text=df.text.apply(lambda x:unidecode(x))
    
    ## Nettoyage
    df.text= df.text.apply(lambda text:nettoyage(text))
    
    return df


############################preprocessing Movies##########################


stopwords_english = read_stopwords("Stopword_alir3z4.txt")
not_stop_words = {'like', 'good', 'bad', 'not', "don't"}
stopwords_english = stopwords_english.difference(not_stop_words)


def preprocess_movies(df_input, stemming=True, spacy_model_size='sm', stop_words=True):
    """Prétraitement du dataset Movies. stemming, lemmatisation, tokenisation, stopwords, minuscules, suppression des caractères non alphabétiques

    Args:
        df_input (dataframe): dataframe des documents originaux
        stemming (bool): Appliquer ou non le stemming (incompatible avec lemmatisation)
        spacy_model_size (str): charger le petit ou le grand modèle de langue de spacy
        stop_words (bool): niveau de suppression des stopwords
        
    Returns:
        dataframe : documents prétraités
    """
    
    df = df_input.copy()
    
    # Charger le modèle de langue si besoin
    
    if not stemming:
        if spacy_model_size == 'lg':
            nlp = spacy.load("en_core_web_lg")
        elif spacy_model_size == 'sm':
            nlp = spacy.load("en_core_web_sm")
            
   
    # Tokenization et lemmatisation et  Suppression des stop words
    
    def tokenization_lemmatization_stopWords(texte):
        if not stemming:
            doc = nlp(texte)
        else:
            doc = nltk.word_tokenize(texte)
        
        tok_lemm_text = []
        
        for token in doc:
            if not stemming : #On fait la lemmatisation
                mot_lemmatised=token.lemma_
        
                if mot_lemmatised not in stopwords_english.union(caracteres_fichier):
                    tok_lemm_text.append(mot_lemmatised)
            
            else: #On fait le stemming
                stemmer = PorterStemmer()
                
                if stop_words:
                    if token not in stopwords_english.union(caracteres_fichier) and (len(token) >= 3) :
                        mot_stemmed = stemmer.stem(token)    
                        tok_lemm_text.append(mot_stemmed)
                elif token not in caracteres_fichier and (len(token) >= 3):
                    mot_stemmed = stemmer.stem(token)    
                    tok_lemm_text.append(mot_stemmed)
            

        return " ".join(tok_lemm_text)
    
    df.text = df.text.apply(lambda text:tokenization_lemmatization_stopWords(text))
    
    ## Transformer les accents en lettres non accentuées  
    df.text = df.text.apply(lambda x:x.lower())  
    df.text = df.text.apply(lambda x:unidecode(x))
    
    ## Nettoyage
    df.text = df.text.apply(lambda text:nettoyage(text))
    
    return df




