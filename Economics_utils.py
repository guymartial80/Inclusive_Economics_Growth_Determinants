import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import time


def importer_affichage_dataset(chemin_fichier):
    """
    Cette fonction importe un fichier Excel ou CSV en indiquant seulement en paramètre le nom du fichier et son extension,
    à condition que ce dernier soit dans le même repertoire que le présent fichier.
    
    Args:
    - chemin_fichier : Nom du fichier et son extension ou Chemin vers le fichier à importer (Excel ou CSV).
    
    Returns:
    - df : DataFrame contenant les données du fichier.
    """
    # Vérifier l'extension du fichier pour déterminer le type de fichier
    if chemin_fichier.endswith('.xlsx'):
        # Importer un fichier Excel
        df = pd.read_excel(chemin_fichier)
    elif chemin_fichier.endswith('.csv'):
        # Importer un fichier CSV
        df = pd.read_csv(chemin_fichier)
    else:
        raise ValueError("Le fichier doit être au format Excel (.xlsx) ou CSV (.csv)")
    
    return df


def count_countries_by_region(data):
    # Compter le nombre de pays par région
    region_counts = data['Region'].value_counts()

    # Créer un DataFrame à partir des comptages
    region_df = region_counts.reset_index()
    region_df.columns = ['Region', 'Nombre de pays']

    # Afficher le tableau
    return region_df



def plot_region_proportion(data):
    # Créer un dictionnaire pour stocker le nombre de pays dans chaque région
    region_counts = {}

    # Définir la palette de couleurs en fonction du nombre unique de régions
    palette = sns.color_palette("husl", len(data['Region'].unique()))

    # Compter le nombre de pays dans chaque région
    for region in data['Region']:
        region_counts[region] = region_counts.get(region, 0) + 1

    # Extraire les données du dictionnaire
    regions = list(region_counts.keys())
    counts = list(region_counts.values())

    # Créer le graphique à barres
    plt.figure(figsize=(10, 6))
    plt.bar(regions, counts, color=palette)
    plt.xlabel('Region')
    plt.ylabel('Nombre de pays')
    plt.title('Proportion de Régions dans chaque pays')
    plt.xticks(rotation=90, ha='right')  # Rotation des étiquettes sur l'axe x
    plt.tight_layout()  # Ajuster la disposition pour éviter que les étiquettes ne se chevauchent
    plt.show()


def plot_bar_chart_region(dataframe, x_column, y_column):
    # Définir la palette de couleurs en fonction du nombre unique de régions
    palette = sns.color_palette("husl", len(dataframe[x_column].unique()))
    
    # Tracer le graphique à barres avec une couleur différente pour chaque région
    dataframe.groupby(x_column)[y_column].sum().sort_values().plot(kind='bar', color=palette)
    plt.ylabel(y_column)
    plt.show()



def plot_bar_chart_gdp(dataframe, x_column, y_column):
    # Définir la palette de couleurs en fonction du nombre unique de régions
    palette = sns.color_palette("husl", len(dataframe[x_column].unique()))
    
    # Tracer le graphique à barres avec une couleur différente pour chaque région
    dataframe.groupby(x_column)[y_column].median().sort_values().plot(kind='bar', color=palette)
    plt.ylabel(y_column)
    plt.show()



def plot_numeric_heatmap(dataframe):
    # Sélectionnons uniquement les colonnes numériques
    numeric_columns = dataframe.select_dtypes(include=['float64', 'int64'])
    
    # Calculons la matrice de corrélations
    corr_matrix = numeric_columns.corr()
    
    # Créeation d'un masque pour partie supérieure de la matrice
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Créer la heatmap des corrélations sans les valeurs de symétrie
    fig, ax = plt.subplots(figsize=(14, 14))
    sns.heatmap(corr_matrix, annot=True, mask=mask, vmin=-1, vmax=1, fmt=".2f", cmap='RdBu')
    plt.show()



def create_regression_pipeline(categorical_vars, numerical_vars, seed=42):
    # Pour les variables catégorielles 
    cat_transform = Pipeline(steps=[
        ('onehot', OneHotEncoder())
    ])

    # Pour les variables numériques
    num_transform = Pipeline(steps=[
        ('sc', StandardScaler())
    ])

    # Créez le ColumnTransformer 
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transform, categorical_vars),  # OneHotEncoder pour les variables catégorielles
            ('num', num_transform, numerical_vars)    # Standardiser pour les variables numériques
        ])

    # Définissez les modèles
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(random_state=seed),
        'XgboostRegressor': XGBRegressor()
    }

    # Créez les pipelines pour chaque modèle
    pipelines = {}
    for model_name, model in models.items():
        pipelines[model_name] = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

    return pipelines



def train_and_evaluate_pipeline(pipelines, X_train, y_train, X_test, y_test):
    results = {}
    metrics = {
        'Model': [],
        'Train MSE': [],
        'Test MSE': [],
        'Train R2': [],
        'Test R2': []
    }
    
    for model_name, pipeline in pipelines.items():
        # Entraîner le pipeline sur les données d'entraînement
        pipeline.fit(X_train, y_train)
        
        # Prédictions sur les données d'entraînement et de test
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Calcul des métriques
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Stocker les résultats
        results[model_name] = {
            'Train MSE': train_mse,
            'Train R2': train_r2,
            'Test MSE': test_mse,
            'Test R2': test_r2
        }
        
        # Ajouter les métriques au tableau
        metrics['Model'].append(model_name)
        metrics['Train MSE'].append(train_mse)
        metrics['Train R2'].append(train_r2)
        metrics['Test MSE'].append(test_mse)
        metrics['Test R2'].append(test_r2)
        
    # Affichage des métriques dans un DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    return metrics_df