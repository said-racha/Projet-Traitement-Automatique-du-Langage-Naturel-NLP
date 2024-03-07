import codecs
import re
import os.path

def load_pres(fname="./datasets/AFDpresidentutf8/corpus.tache1.learn.utf8.txt"):
    """Charge le dataset de train Président

    Args:
        fname (str, optional): Chemin vers le fichier texte contenant le dataset.

    Returns:
        alltxts: liste de tous les documents
        alllabs: liste de tous les labels
    """
    alltxts = []
    alllabs = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        #
        lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
        txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
        if lab.count('M') >0:
            alllabs.append(-1)
        else: 
            alllabs.append(1)
        alltxts.append(txt)
    return alltxts,alllabs

def load_test_pres(fname="datasets/AFDpresidentutf8/corpus.tache1.test.utf8.txt"):
    """Charge le dataset de test President

    Args:
        fname (str, optional): chemin vers le fichier texte.

    Returns:
        alltxts (list[str]): la liste des documents
    """
    alltxts = []
    s=codecs.open(fname, 'r','utf-8') # pour régler le codage
    while True:
        txt = s.readline()
        if(len(txt))<5:
            break
        
        txt = re.sub(r"<[0-9]*:[0-9]*>(.*)","\\1",txt)
        
        alltxts.append(txt)
    return alltxts

def load_movies(path2data="./datasets/movies1000/"): 
    """Charge le dataset de train de Movies.

    Args:
        path2data (str, optional): Chemin vers le dossier racine contenant 
        les reviews répertoires des reviews positives et négatives

    Returns:
        alltxts: liste de tous les documents
        alllabs: liste de tous les labels
    """
    alltxts = [] # init vide
    labs = []
    cpt = 0
    for cl in os.listdir(path2data): # parcours des fichiers d'un répertoire
        for f in os.listdir(path2data+cl):
            txt = open(path2data+cl+'/'+f).read()
            alltxts.append(txt)
            labs.append(cpt)
        cpt+=1 # chg répertoire = cht classe
        
    return alltxts,labs

def load_test_movies(path2data="./datasets/testSentiment.txt"):
    """Charge le dataset de test de Movies

    Args:
        path2data (str, optional): 

    Returns:
        _type_: _description_
    """
    alltxts = []
    
    with open(path2data) as file:
        for line in file:
            alltxts.append(line)
            
    return alltxts
  
