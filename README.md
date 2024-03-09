# Présentation
Projet de la partie Traitement Automatique du Langage de l'UE RITAL (parcours DAC) M1-S2 à Sorbonne Université 

# Binôme
- MAOUCHE Mounir M1-IMA
- [SAID Racha](https://github.com/said-racha) M1-DAC

# Problèmes étudiés
- Reconnaissance de locuteur
- Analyse de sentiments

# Principe
- Utilisation d'une représentation bag of words pour de la classification de textes sur deux datasets différents
- Détermination d'une stratégie de nettoyage et de normalisation des textes adaptée pour chacune des tâches
- Elaboration d'un système de tests permettant de sélectionner les modèles les plus performants ainsi que l'affinage de leurs paramètres
- Evaluation des performances des modèles à travers des mesures adaptées à la nature des datasets (equilibré/déséquilibré)
- Etude et utilisation de différentes notions de machine learning pour maximiser le score du modèle (sur/sous-échantillonnage, lissage, régularisation...)

# Datasets:
- **"Présidents"** : Phrases extraites d'un débat entre François Mitterrand et Jacques Chirac
- **"Movies"** : Revues de films accompagnées de leur polarité
 
# Performances et résultats
## Dataset "Présidents" :
- **f1-score** : 71.68 %
- **AUC ROC** : 96.43 %
- **AUC RP** : 87.23 %

## Dataset "Movies" :
- **accuracy** : 81.16 % 
- **précision**: 89.16 %
- **rappel** : 81.44 %
