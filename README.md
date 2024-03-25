# Indicateurs du développement économique

## SOMMAIRE

- [DESCRIPTION DU PROJET](#1.-DESCRIPTION)
- [ETAPES DE REALISATION](#2.-ETAPES-DE-REALISATION)
- [LIBRAIRIES UTILISEES](#3.-LIBRAIRIES-UTILISEES)

## 1. DESCRIPTION

Les données proviennent de 227 pays et contiennent des informations sur certains des facteurs importants qui gouvernent le développement économique. Il s'agit de modéliser lesdites données dans l'optique de prédire en fonction de certains indicateurs économiques d'une région, le Produit Intérieur Brut (PIB) par habitant.

## 2. ETAPES DE REALISATION

### 2.1 Importation des librairies
Les packages de base ont été importés dans un premier temps, notamment Pandas, numpy, Matplolib, Seaborn.

### 2.2 Structure globale du travail
Pour une meilleure lisibilité du code principal, le fichier "Economics_utils.py" a été créé pour héberger toutes les fonctions nécessaires pour ce projet. Ces dernières y sont d'ailleurs décrites. Ainsi, lesdites fonctions sont appelées dans le fichier principal "Measures_Economics.ipynb" pour exécution.

### 2.3 Importation et structure du dataset
#### 2.3.1 Importation et aperçu
Le jeu de données est importé à travers la fonction implémentée à cet effet; l'importation est faite pour s'assurer de la réussite de l'opération.

#### 2.3.2 Structure du dataset
Cette partie nous permet d'en savoir un peu plus sur le contenu de la base de données, tout en sachant que la variable cible est GDP ($ per capita).

#### 2.3.3 Résumé statistique
Calculs statistiques de base sur toutes les variables de type numérique du jeu de données.

### 2.4 EDA
A cette étape, plusieurs démarches ont été sollicitées pour faciliter l'analyse :
- Les statistiques univariées: Recueillir le nombre d'occurrences par variables de type catégoriel;
- Les statistiques bivariées : Nombre d'occurrences de chaque variable catégorielle en fonction des modalités de la variable cible;
- Les statistiques bivariées pour les variables  de type numérique : cela s'est fait par la matrice des corrélations et les boites à moustache.

### 2.5 Machine Learning
- Séparation des variables (explicatives et expliquée);
- Séparation en base d'entrainement et test;
- Modélisation, et évaluation de l'algorithme.

### 2.6 Résultats obtenus
Le processus de modélisation a revélé, après évaluation de trois modèles de Machine Learning (Regression linéaire multiple, Random Forest regressor et Xgboost regressor),  nous obtenons les résultats suivants  :
  |Modèles|Train MSE|Train R2|Test MSE|Test R2|
  |-------|---------|--------|--------|-------|
  |LinearRegression|2.005836e+07|3.267530e+07|0.803437|0.6591|
  |RandomForestRegressor|3.408791e+06|8.134701e+06|0.966595|0.9151|
  |XgboostRegressor|1.053881e-03|1.736429e+07|1.000000|0.8188|

## 3. LIBRAIRIES UTILISEES
![Static Badge](https://img.shields.io/badge/Pandas-black?style=for-the-badge&logo=Pandas) ![Static Badge](https://img.shields.io/badge/Scikit-learn-black?style=for-the-badge&logo=Scikit-learn) ![Static Badge](https://img.shields.io/badge/Numpy-black?style=for-the-badge&logo=Numpy) ![Static Badge](https://img.shields.io/badge/Matplotlib-black?style=for-the-badge&logo=Matplotlib) ![Static Badge](https://img.shields.io/badge/Seaborn-black?style=for-the-badge&logo=Seaborn)


