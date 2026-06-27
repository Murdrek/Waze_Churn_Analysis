#!/usr/bin/env python
# coding: utf-8

# # **Waze Project**
# **Course 5 - The nuts and bolts of machine learning**

# Your team is close to completing their user churn project. Previously, you completed a project proposal, and used Python to explore and analyze Waze’s user data, create data visualizations, and conduct a hypothesis test. Most recently, you built a binomial logistic regression model based on multiple variables.
# 
# Leadership appreciates all your hard work. Now, they want your team to **build a machine learning model to predict user churn**. To get the best results, your team decides to build and test two tree-based models: random forest and XGBoost.
# 
# Your work will help leadership make informed business decisions to prevent user churn, improve user retention, and grow Waze’s business.
# 

# # **Course 6 End-of-Course Project: Build a machine learning model**
# 
# In this activity, you will practice using tree-based modeling techniques to predict on a binary target class.
# <br/>
# 
# **The purpose** of this model is to find factors that drive user churn.
# 
# **The goal** of this model is to predict whether or not a Waze user is retained or churned.
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Ethical considerations
# * Consider the ethical implications of the request
# 
# * Should the objective of the model be adjusted?
# 
# **Part 2:** Feature engineering
# 
# * Perform feature selection, extraction, and transformation to prepare the data for modeling
# 
# **Part 3:** Modeling
# 
# * Build the models, evaluate them, and advise on next steps
# 
# Follow the instructions and answer the questions below to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.
# 
# ES: 
# 
# Proyecto Waze
# Curso 5 - Los componentes fundamentales del aprendizaje automático (The nuts and bolts of machine learning)
# Su equipo está cerca de completar el proyecto sobre la pérdida de usuarios (user churn). Anteriormente, completaron una propuesta de proyecto y utilizaron Python para explorar y analizar los datos de los usuarios de Waze, crear visualizaciones de datos y realizar una prueba de hipótesis. Más recientemente, construyeron un modelo de regresión logística binomial basado en múltiples variables.
# 
# La dirección agradece todo su arduo trabajo. Ahora, quieren que su equipo construya un modelo de aprendizaje automático para predecir la pérdida de usuarios. Para obtener los mejores resultados, su equipo ha decidido construir y probar dos modelos basados en árboles: Random Forest (Bosque Aleatorio) y XGBoost.
# 
# Su trabajo ayudará a la dirección a tomar decisiones comerciales informadas para prevenir la pérdida de usuarios, mejorar la retención de los mismos y hacer crecer el negocio de Waze.
# 
# Proyecto de fin de curso del Curso 5: Construir un modelo de aprendizaje automático
# En esta actividad, practicará el uso de técnicas de modelado basadas en árboles para realizar predicciones sobre una clase objetivo binaria.
# 
# El propósito de este modelo es encontrar los factores que impulsan la pérdida de usuarios.
# 
# El objetivo de este modelo es predecir si un usuario de Waze se mantiene activo (retained) o si abandona la aplicación (churned).
# 
# Esta actividad consta de tres partes:
# 
# Parte 1: Consideraciones éticas
# Evaluar las implicaciones éticas de la solicitud.
# 
# ¿Debería ajustarse el objetivo del modelo?
# 
# Parte 2: Ingeniería de características (Feature Engineering)
# Realizar la selección, extracción y transformación de variables para preparar los datos para el modelado.
# 
# Parte 3: Modelado
# Construir los modelos, evaluarlos y asesorar sobre los siguientes pasos.
# 
# Siga las instrucciones y responda a las preguntas que se presentan a continuación para completar la actividad. Luego, completará un Resumen Ejecutivo utilizando las preguntas enumeradas en el Documento de Estrategia PACE.
# 
# Asegúrese de completar esta actividad antes de avanzar. El próximo elemento del curso le proporcionará un modelo ejemplar ya completado para que pueda compararlo con su propio trabajo.

# # **Build a machine learning model**
# 

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**
# 

# Throughout these project notebooks, you'll see references to the problem-solving framework PACE. The following notebook components are labeled with the respective PACE stage: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following questions:
# 
# 1.   What are you being asked to do?
# EN:
# 
# You are being asked to build and evaluate two tree-based machine learning models (Random Forest and XGBoost) to predict a binary target class: whether a Waze user will be retained or churned.
# 
# ES:
# 
# Se te pide que construyas y evalúes dos modelos de aprendizaje automático basados en árboles (Random Forest y XGBoost) para predecir una clase objetivo binaria: si un usuario de Waze se mantendrá activo (retained) o abandonará la app (churned).
# 
# 2.   What are the ethical implications of the model? What are the consequences of your model making errors?
# EN:
# 
# Ethical Implications & Error Consequences: The model uses behavioral driving data, which requires strict anonymity to prevent tracking real-world individuals. Errors do not cause physical harm, but they lead to inefficient resource allocation and poor user experience.
# 
# ES:
# Implicaciones Éticas y Consecuencias de Errores: El modelo utiliza datos de comportamiento de conducción, lo que exige un estricto anonimato para evitar el rastreo de personas reales. Los errores no causan daños físicos, pero provocan una asignación ineficiente de recursos y una mala experiencia de usuario.
# 
#   *   What is the likely effect of the model when it predicts a false negative (i.e., when the model says a Waze user won't churn, but they actually will)?
#   
#   EN:
#   
#   False Negative Effect: Waze fails to identify a user at risk of leaving. No retention offer or notification is sent, resulting in the silent loss of a customer to competitors.
# 
# 
# 
# ES:
# 
# Efecto de un Falso Negativo: Waze no identifica a un usuario en riesgo de irse. No se envía ninguna oferta de retención, lo que resulta en la pérdida silenciosa de un cliente frente a la competencia.
# 
# 
# 
#   
#   *   What is the likely effect of the model when it predicts a false positive (i.e., when the model says a Waze user will churn, but they actually won't)?
#   EN:
#   
#   False Positive Effect: Waze flags a loyal user as someone about to leave. This leads to wasted marketing budget by sending unnecessary incentives or promotions to someone who was going to stay anyway.
# 
# ES:
# 
# Efecto de un Falso Positivo: Waze etiqueta a un usuario leal como alguien a punto de irse. Esto genera un desperdicio del presupuesto de marketing al enviar incentivos o promociones innecesarias a alguien que se iba a quedar de todos modos.
# 
# 3.  Do the benefits of such a model outweigh the potential problems?
# 
# EN:
# 
# Do Benefits Outweigh Problems? Yes. Since this is a commercial retention tool, the risks are financial and operational, not life-threatening or discriminatory. Modest optimization easily manages the costs of misclassification.
# 
# 
# es:
# 
# ¿Los beneficios superan los problemas? Sí. Al ser una herramienta comercial de retención, los riesgos son financieros y operativos, no vitales ni discriminatorios. Una optimización moderada maneja fácilmente los costos de una clasificación errónea.
# 
# 4.  Would you proceed with the request to build this model? Why or why not?
# 
# EN:
# 
# Would you proceed? Yes. The objective is ethically sound. It improves user experience by identifying pain points in the app and drives sustainable business growth.
# 
# ES:
# 
# ¿Procederías con el proyecto? Sí. El objetivo es éticamente correcto. Permite mejorar la experiencia del usuario al identificar fallas en la aplicación y fomenta el crecimiento sostenible del negocio.

# ==> ENTER YOUR RESPONSES TO QUESTIONS 1-4 HERE

# ### **Task 1. Imports and data loading**
# 
# Import packages and libraries needed to build and evaluate random forest and XGBoost classification models.

# In[1]:


# ==========================================
# Task 1. Imports and data loading
# ==========================================

# Import packages for data manipulation / Importación para manipulación de datos
import numpy as np
import pandas as pd

# Import packages for data visualization / Importación para visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

# This lets us see all of the columns, preventing Jupyter from redacting them.
# Esto permite ver todas las columnas, evitando que Jupyter las oculte.
pd.set_option('display.max_columns', None)

# Import packages for data modeling / Importación para modelado de datos
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# This is the function that helps plot feature importance
# Función para graficar la importancia de las variables en XGBoost
from xgboost import plot_importance

# This module lets us save our models once we fit them.
# Módulo para guardar los modelos una vez entrenados.
import pickle


# Now read in the dataset as `df0` and inspect the first five rows.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Import dataset
df0 = pd.read_csv('waze_dataset.csv')


# In[3]:


# Inspect the first five rows
### YOUR CODE HERE ###
# Inspeccionar las primeras cinco filas del conjunto de datos
df0.head()


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2. Feature engineering**
# 
# You have already prepared much of this data and performed exploratory data analysis (EDA) in previous courses. You know that some features had stronger correlations with churn than others, and you also created some features that may be useful.
# 
# In this part of the project, you'll engineer these features and some new features to use for modeling.
# 
# To begin, create a copy of `df0` to preserve the original dataframe. Call the copy `df`.

# In[4]:


# Copy the df0 dataframe
### YOUR CODE HERE ###
df = df0.copy()


# Call `info()` on the new dataframe so the existing columns can be easily referenced.

# In[5]:


### YOUR CODE HERE ###
# Llamar a info() en el nuevo dataframe para inspeccionar columnas y tipos de datos
df.info()


# #### **`km_per_driving_day`**
# 
# 1. Create a feature representing the mean number of kilometers driven on each driving day in the last month for each user. Add this feature as a column to `df`.
# 
# 2. Get descriptive statistics for this new feature
# 
# 

# In[6]:


# 1. Create `km_per_driving_day` feature
### YOUR CODE HERE ###
# 1. Crear la variable `km_per_driving_day` usando el esquema exacto de tu dataset
# kilómetros recorridos = driven_km_drives | días_conduciendo = driving_days
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# 2. Get descriptive stats
### YOUR CODE HERE ###
# 2. Obtener estadísticas descriptivas
df['km_per_driving_day'].describe()


# Notice that some values are infinite. This is the result of there being values of zero in the `driving_days` column. Pandas imputes a value of infinity in the corresponding rows of the new column because division by zero is undefined.
# 
# 1. Convert these values from infinity to zero. You can use `np.inf` to refer to a value of infinity.
# 
# 2. Call `describe()` on the `km_per_driving_day` column to verify that it worked.

# In[7]:


# 1. Convert infinite values to zero
### YOUR CODE HERE ###

# 1. Convertir los valores infinitos a cero
df['km_per_driving_day'] = df['km_per_driving_day'].replace([np.inf, -np.inf], 0)

# 2. Confirm that it worked
### YOUR CODE HERE ###
# 2. Confirmar que funcionó correctamente
df['km_per_driving_day'].describe()


# #### **`percent_sessions_in_last_month`**
# 
# 1. Create a new column `percent_sessions_in_last_month` that represents the percentage of each user's total sessions that were logged in their last month of use.
# 
# 2. Get descriptive statistics for this new feature

# In[8]:


# 1. Create `percent_sessions_in_last_month` feature
### YOUR CODE HERE ###
# 1. Crear la variable `percent_sessions_in_last_month`
df['percent_sessions_in_last_month'] = df['sessions'] / df['total_sessions']

# 1. Get descriptive stats
### YOUR CODE HERE ###
# 2. Obtener estadísticas descriptivas
df['percent_sessions_in_last_month'].describe()


# #### **`professional_driver`**
# 
# Create a new, binary feature called `professional_driver` that is a 1 for users who had 60 or more drives <u>**and**</u> drove on 15+ days in the last month.
# 
# **Note:** The objective is to create a new feature that separates professional drivers from other drivers. In this scenario, domain knowledge and intuition are used to determine these deciding thresholds, but ultimately they are arbitrary.

# To create this column, use the [`np.where()`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) function. This function accepts as arguments:
# 1. A condition
# 2. What to return when the condition is true
# 3. What to return when the condition is false
# 
# ```
# Example:
# x = [1, 2, 3]
# x = np.where(x > 2, 100, 0)
# x
# array([  0,   0, 100])
# ```
# ES:
# 
# conductor_profesional / professional_driver
# 
# Cree una nueva variable binaria llamada professional_driver que tome el valor de 1 para los usuarios que hayan realizado 60 o más viajes (drives) y que además hayan conducido durante 15 o más días (driving_days) en el último mes.
# 
# Nota: El objetivo es crear una nueva característica que separe a los conductores profesionales del resto de los conductores. En este escenario, se utiliza el conocimiento del dominio y la intuición para determinar estos umbrales de decisión, pero en última instancia son arbitrarios.
# 
# Para crear esta columna, utilice la función np.where() de NumPy. Esta función acepta como argumentos:
# 
# Una condición lógica.
# 
# Qué devolver cuando la condición es verdadera (True).
# 
# Qué devolver cuando la condición es falsa (False).

# In[9]:


# Create `professional_driver` feature
### YOUR CODE HERE ###
# Crear la variable `professional_driver` basada en la condición conjunta
df['professional_driver'] = np.where(
    (df['drives'] >= 60) & (df['driving_days'] >= 15), 
    1, 
    0
)

# Get the count of each class to inspect the distribution
# Obtener el recuento de cada clase para inspeccionar la distribución
df['professional_driver'].value_counts()


# #### **`total_sessions_per_day`**
# 
# Now, create a new column that represents the mean number of sessions per day _since onboarding_.

# In[10]:


# Create `total_sessions_per_day` feature
### YOUR CODE HERE ###
# Crear la variable `total_sessions_per_day`
df['total_sessions_per_day'] = df['total_sessions'] / df['n_days_after_onboarding']


# As with other features, get descriptive statistics for this new feature.

# In[11]:


# Get descriptive stats
### YOUR CODE HERE ###
# Obtener estadísticas descriptivas para auditar el resultado
df['total_sessions_per_day'].describe()


# #### **`km_per_hour`**
# 
# Create a column representing the mean kilometers per hour driven in the last month.

# In[12]:


# Create `km_per_hour` feature
### YOUR CODE HERE ###
# Crear la variable `km_per_hour`
df['km_per_hour'] = df['driven_km_drives'] / (df['duration_minutes_drives'] / 60)
# Print descriptive statistics for the feature
# Imprimir estadísticas descriptivas para la variable
df['km_per_hour'].describe()


# #### **`km_per_drive`**
# 
# Create a column representing the mean number of kilometers per drive made in the last month for each user. Then, print descriptive statistics for the feature.

# In[13]:


# Create `km_per_drive` feature
### YOUR CODE HERE ###
df['km_per_drive'] = df['driven_km_drives'] / df['drives']
# Print descriptive statistics for the feature
# Imprimir estadísticas descriptivas para la variable
df['km_per_drive'].describe()


# This feature has infinite values too. Convert the infinite values to zero, then confirm that it worked.

# In[14]:


# 1. Convert infinite values to zero
### YOUR CODE HERE ###
# 1. Convertir los valores infinitos a cero
df['km_per_drive'] = df['km_per_drive'].replace([np.inf, -np.inf], 0)
# 2. Confirm that it worked
### YOUR CODE HERE ###
# 2. Confirmar que funcionó correctamente
df['km_per_drive'].describe()


# #### **`percent_of_sessions_to_favorite`**
# 
# Finally, create a new column that represents the percentage of total sessions that were used to navigate to one of the users' favorite places. Then, print descriptive statistics for the new column.
# 
# This is a proxy representation for the percent of overall drives that are to a favorite place. Since total drives since onboarding are not contained in this dataset, total sessions must serve as a reasonable approximation.
# 
# People whose drives to non-favorite places make up a higher percentage of their total drives might be less likely to churn, since they're making more drives to less familiar places.

# In[15]:


# Create `percent_of_sessions_to_favorite` feature
### YOUR CODE HERE ###
# Crear la variable `percent_of_sessions_to_favorite`
df['percent_of_sessions_to_favorite'] = (df['total_navigations_fav1'] + df['total_navigations_fav2']) / df['total_sessions']

# Get descriptive stats
### YOUR CODE HERE ###
# Obtener estadísticas descriptivas para auditar la distribución
df['percent_of_sessions_to_favorite'].describe()


# ### **Task 3. Drop missing values**
# 
# Because you know from previous EDA that there is no evidence of a non-random cause of the 700 missing values in the `label` column, and because these observations comprise less than 5% of the data, use the `dropna()` method to drop the rows that are missing this data.

# In[16]:


# Drop rows with missing values
### YOUR CODE HERE ### 
# Eliminar filas con valores faltantes en la columna 'label'
df = df.dropna(subset=['label'])

# Verify that there are no more missing values in 'label'
# Verificar que ya no queden valores faltantes en 'label'
df['label'].isna().sum()


# ### **Task 4. Outliers**
# 
# You know from previous EDA that many of these columns have outliers. However, tree-based models are resilient to outliers, so there is no need to make any imputations.

# ### **Task 5. Variable encoding**

# #### **Dummying features**
# 
# In order to use `device` as an X variable, you will need to convert it to binary, since this variable is categorical.
# 
# In cases where the data contains many categorical variables, you can use pandas built-in [`pd.get_dummies()`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), or you can use scikit-learn's [`OneHotEncoder()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) function.
# 
# **Note:** Each possible category of each feature will result in a feature for your model, which could lead to an inadequate ratio of features to observations and/or difficulty understanding your model's predictions.
# 
# Because this dataset only has one remaining categorical feature (`device`), it's not necessary to use one of these special functions. You can just implement the transformation directly.
# 
# Create a new, binary column called `device2` that encodes user devices as follows:
# 
# * `Android` -> `0`
# * `iPhone` -> `1`

# In[17]:


# Create new `device2` variable
### YOUR CODE HERE ###
# Crear la nueva variable binaria `device2` usando np.where
# Android será 0 e iPhone será 1
df['device2'] = np.where(df['device'] == 'iPhone', 1, 0)

# Verify the transformation with the first few rows
# Verificar la transformación imprimiendo las primeras filas de ambas columnas
df[['device', 'device2']].head(10)


# #### **Target encoding**
# 
# The target variable is also categorical, since a user is labeled as either "churned" or "retained." Change the data type of the `label` column to be binary. This change is needed to train the models.
# 
# Assign a `0` for all `retained` users.
# 
# Assign a `1` for all `churned` users.
# 
# Save this variable as `label2` so as not to overwrite the original `label` variable.
# 
# **Note:** There are many ways to do this. Consider using `np.where()` as you did earlier in this notebook.

# In[18]:


# Create binary `label2` column
### YOUR CODE HERE ###
# Crear la columna binaria `label2` usando np.where
# 'retained' será 0 y 'churned' será 1
df['label2'] = np.where(df['label'] == 'churned', 1, 0)

# Verify the transformation with a value count
# Verificar la transformación con un recuento de valores para asegurar el mapeo
df[['label', 'label2']].value_counts()


# ### **Task 6. Feature selection**
# 
# Tree-based models can handle multicollinearity, so the only feature that can be cut is `ID`, since it doesn't contain any information relevant to churn.
# 
# Note, however, that `device` won't be used simply because it's a copy of `device2`.
# 
# Drop `ID` from the `df` dataframe.

# In[19]:


# Drop `ID` column
### YOUR CODE HERE ###
# Eliminar la columna `ID` del dataframe
df = df.drop(columns=['ID'])

# Verify that 'ID' was successfully dropped by checking the columns
# Verificar que 'ID' fue eliminado correctamente revisando las columnas actuales
df.info()


# ### **Task 7. Evaluation metric**
# 
# Before modeling, you must decide on an evaluation metric. This will depend on the class balance of the target variable and the use case of the model.
# 
# First, examine the class balance of your target variable.

# In[20]:


# Get class balance of 'label' col
### YOUR CODE HERE ###
# Obtener el balance de clases de la columna 'label2' en porcentaje/proporción
df['label2'].value_counts(normalize=True)


# Approximately 18% of the users in this dataset churned. This is an unbalanced dataset, but not extremely so. It can be modeled without any class rebalancing.
# 
# Now, consider which evaluation metric is best. Remember, accuracy might not be the best gauge of performance because a model can have high accuracy on an imbalanced dataset and still fail to predict the minority class.
# 
# It was already determined that the risks involved in making a false positive prediction are minimal. No one stands to get hurt, lose money, or suffer any other significant consequence if they are predicted to churn. Therefore, select the model based on the recall score.

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# ### **Task 8. Modeling workflow and model selection process**
# 
# The final modeling dataset contains 14,299 samples. This is towards the lower end of what might be considered sufficient to conduct a robust model selection process, but still doable.
# 
# 1. Split the data into train/validation/test sets (60/20/20)
# 
# Note that, when deciding the split ratio and whether or not to use a validation set to select a champion model, consider both how many samples will be in each data partition, and how many examples of the minority class each would therefore contain. In this case, a 60/20/20 split would result in \~2,860 samples in the validation set and the same number in the test set, of which \~18%&mdash;or 515 samples&mdash;would represent users who churn.
# 2. Fit models and tune hyperparameters on the training set
# 3. Perform final model selection on the validation set
# 4. Assess the champion model's performance on the test set
# 
# ![](https://raw.githubusercontent.com/adacert/tiktok/main/optimal_model_flow_numbered.svg)

# ### **Task 9. Split the data**
# 
# Now you're ready to model. The only remaining step is to split the data into features/target variable and training/validation/test sets.
# 
# 1. Define a variable `X` that isolates the features. Remember not to use `device`.
# 
# 2. Define a variable `y` that isolates the target variable (`label2`).
# 
# 3. Split the data 80/20 into an interim training set and a test set. Don't forget to stratify the splits, and set the random state to 42.
# 
# 4. Split the interim training set 75/25 into a training set and a validation set, yielding a final ratio of 60/20/20 for training/validation/test sets. Again, don't forget to stratify the splits and set the random state.

# In[21]:


# 1. Isolate X variables
### YOUR CODE HERE ###
# Aislar las variables predictoras X (excluyendo texto antiguo y variables objetivo)
X = df.drop(columns=['label', 'label2', 'device'])

# 2. Isolate y variable
### YOUR CODE HERE ###
# Aislar la variable objetivo y
y = df['label2']

# 3. Split into train and test sets
### YOUR CODE HERE ###
# Dividir en conjunto intermedio (entrenamiento/validación) y prueba (80/20)
from sklearn.model_selection import train_test_split

X_interim, X_test, y_interim, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 4. Split into train and validate sets
### YOUR CODE HERE ###

# Dividir el conjunto intermedio en entrenamiento y validación final (75/25)
X_train, X_val, y_train, y_val = train_test_split(
    X_interim, y_interim, test_size=0.25, stratify=y_interim, random_state=42
)


# Verify the number of samples in the partitioned data.

# In[22]:


### YOUR CODE HERE ###
# Verify the dimensions (number of rows and columns) for each partition
# Verificar las dimensiones (número de filas y columnas) para cada partición
print("--- Data Partitions Dimensions / Dimensiones de las Particiones ---")
print(f"X_train shape:      {X_train.shape}")
print(f"X_val shape:        {X_val.shape}")
print(f"X_test shape:       {X_test.shape}\n")

# Verify class balance stratification across all sets
# Verificar la estratificación del balance de clases en todos los conjuntos
print("--- Target Class Balance / Balance de la Clase Objetivo ---")
print("Train Set Proportions:\n", y_train.value_counts(normalize=True), "\n")
print("Validation Set Proportions:\n", y_val.value_counts(normalize=True), "\n")
print("Test Set Proportions:\n", y_test.value_counts(normalize=True))


# This aligns with expectations.

# ### **Task 10. Modeling**

# #### **Random forest**
# 
# Begin with using `GridSearchCV` to tune a random forest model.
# 
# 1. Instantiate the random forest classifier `rf` and set the random state.
# 
# 2. Create a dictionary `cv_params` of any of the following hyperparameters and their corresponding values to tune. The more you tune, the better your model will fit the data, but the longer it will take.
#  - `max_depth`
#  - `max_features`
#  - `max_samples`
#  - `min_samples_leaf`
#  - `min_samples_split`
#  - `n_estimators`
# 
# 3. Define a list `scoring` of scoring metrics for GridSearch to capture (precision, recall, F1 score, and accuracy).
# 
# 4. Instantiate the `GridSearchCV` object `rf_cv`. Pass to it as arguments:
#  - estimator=`rf`
#  - param_grid=`cv_params`
#  - scoring=`scoring`
#  - cv: define the number of cross-validation folds you want (`cv=_`)
#  - refit: indicate which evaluation metric you want to use to select the model (`refit=_`)
# 
#  `refit` should be set to `'recall'`.<font/>
# 

# **Note:** If your model fitting takes too long, try reducing the number of options to search over in the grid search.

# In[23]:


# 1. Instantiate the random forest classifier
### YOUR CODE HERE ###
# Instanciar el clasificador de Random Forest fijando el random_state para replicabilidad
rf = RandomForestClassifier(random_state=42)

# 2. Create a dictionary of hyperparameters to tune
### YOUR CODE HERE ###
# Crear el diccionario de hiperparámetros con opciones estratégicas para evitar sobreajuste
cv_params = {
    'max_depth': [4, 6, 8],
    'max_features': ['sqrt'],
    'max_samples': [0.7, 0.9],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 4],
    'n_estimators': [100, 150]
}

# 3. Define a list of scoring metrics to capture
### YOUR CODE HERE ###
# Definir las métricas de evaluación. Nota: Scikit-learn usa strings específicos para cada una
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
### YOUR CODE HERE ###

# Instanciar GridSearchCV buscando optimizar específicamente el 'recall'
rf_cv = GridSearchCV(
    estimator=rf,
    param_grid=cv_params,
    scoring=scoring,
    cv=4,
    refit='recall',
    n_jobs=-1  # Usa todos los núcleos del procesador en paralelo
)


# Now fit the model to the training data.

# In[24]:


### YOUR CODE HERE ###
# Ajustar (entrenar) el objeto GridSearchCV con los datos de entrenamiento
rf_cv.fit(X_train, y_train)


# Examine the best average score across all the validation folds.

# In[25]:


# Examine best score
### YOUR CODE HERE ###
# Examinar el mejor puntaje promedio (Recall) a través de los pliegues de validación
rf_cv.best_score_


# Examine the best combination of hyperparameters.

# In[26]:


# Examine best hyperparameter combo
### YOUR CODE HERE ###
# Examinar la mejor combinación de hiperparámetros encontrada por GridSearchCV
rf_cv.best_params_


# Use the `make_results()` function to output all of the scores of your model. Note that the function accepts three arguments.
# 
# This function is provided for you, but if you'd like to challenge yourself, try writing your own function!

# <details>
#   <summary><h5>HINT</h5></summary>
# 
# To learn more about how this function accesses the cross-validation results, refer to the [`GridSearchCV` scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV) for the `cv_results_` attribute.
# 
# </details>

# In[27]:


def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, or accuracy

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy',
                   }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          },
                         )

    return table


# Pass the `GridSearch` object to the `make_results()` function.

# In[28]:


### YOUR CODE HERE ###
# Llamar a la función para mostrar la tabla de resultados de validación cruzada del Random Forest
make_results('Random Forest CV', rf_cv, 'recall')


# Asside from the accuracy, the scores aren't that good. However, recall that when you built the logistic regression model in the last course the recall was \~0.09, which means that this model has 33% better recall and about the same accuracy, and it was trained on less data.
# 
# If you want, feel free to try retuning your hyperparameters to try to get a better score. You might be able to marginally improve the model.

# #### **XGBoost**
# 
#  Try to improve your scores using an XGBoost model.
# 
# 1. Instantiate the XGBoost classifier `xgb` and set `objective='binary:logistic'`. Also set the random state.
# 
# 2. Create a dictionary `cv_params` of the following hyperparameters and their corresponding values to tune:
#  - `max_depth`
#  - `min_child_weight`
#  - `learning_rate`
#  - `n_estimators`
# 
# 3. Define a list `scoring` of scoring metrics for grid search to capture (precision, recall, F1 score, and accuracy).
# 
# 4. Instantiate the `GridSearchCV` object `xgb_cv`. Pass to it as arguments:
#  - estimator=`xgb`
#  - param_grid=`cv_params`
#  - scoring=`scoring`
#  - cv: define the number of cross-validation folds you want (`cv=_`)
#  - refit: indicate which evaluation metric you want to use to select the model (`refit='recall'`)

# In[29]:


# 1. Instantiate the XGBoost classifier
### YOUR CODE HERE ###
# Instanciar el clasificador XGBoost configurando el objetivo logístico y el estado aleatorio
xgb = XGBClassifier(objective='binary:logistic', random_state=42)

# 2. Create a dictionary of hyperparameters to tune
### YOUR CODE HERE ###
# Crear la grilla de hiperparámetros optimizada para balancear sesgo y varianza
cv_params = {
    'max_depth': [4, 6],
    'min_child_weight': [1, 2],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 150]
}

# 3. Define a list of scoring metrics to capture
### YOUR CODE HERE ###
# Lista de métricas idéntica a la del Random Forest para mantener una comparación justa
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
### YOUR CODE HERE ###
# Configurar la grilla manteniendo cv=4 y refit apuntando estrictamente al 'recall'
xgb_cv = GridSearchCV(
    estimator=xgb,
    param_grid=cv_params,
    scoring=scoring,
    cv=4,
    refit='recall',
    n_jobs=-1  # Ejecución en paralelo usando todos los núcleos del procesador
)


# Now fit the model to the `X_train` and `y_train` data.
# 
# Note this cell might take several minutes to run.

# In[30]:


### YOUR CODE HERE ###
# Ajustar (entrenar) el objeto GridSearchCV de XGBoost con los datos de entrenamiento
xgb_cv.fit(X_train, y_train)


# Get the best score from this model.

# In[31]:


# Examine best score
### YOUR CODE HERE ###
# Examinar el mejor puntaje promedio (Recall) a través de los pliegues de validación para XGBoost
xgb_cv.best_score_


# And the best parameters.

# In[32]:


# Examine best parameters
### YOUR CODE HERE ###
# Examinar la mejor combinación de hiperparámetros encontrada por GridSearchCV para XGBoost
xgb_cv.best_params_


# Use the `make_results()` function to output all of the scores of your model. Note that the function accepts three arguments.

# In[33]:


# Call 'make_results()' on the GridSearch object
### YOUR CODE HERE ###
# Llamar a la función para mostrar la tabla de resultados de validación cruzada de XGBoost
make_results('XGBoost CV', xgb_cv, 'recall')


# This model fit the data even better than the random forest model. The recall score is nearly double the recall score from the logistic regression model from the previous course, and it's almost 50% better than the random forest model's recall score, while maintaining a similar accuracy and precision score.

# ### **Task 11. Model selection**
# 
# Now, use the best random forest model and the best XGBoost model to predict on the validation data. Whichever performs better will be selected as the champion model.

# #### **Random forest**

# In[35]:


# Use random forest model to predict on validation data
### YOUR CODE HERE ###
# Usar el mejor modelo de bosque aleatorio para predecir sobre los datos de validación
rf_val_preds = rf_cv.predict(X_val)


# Use the `get_test_scores()` function to generate a table of scores from the predictions on the validation data.

# In[37]:


def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
        model_name (string): Your choice: how the model will be named in the output table
        preds: numpy array of test predictions
        y_test_data: numpy array of y_test data

    Out:
        table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy]
                          })

    return table


# In[38]:


# Get validation scores for RF model
### YOUR CODE HERE ###
# Obtener las predicciones y calcular las métricas usando los datos de validación
rf_val_preds = rf_cv.predict(X_val)
rf_val_scores = get_test_scores('Random Forest Val', rf_val_preds, y_val)

# Append to the results table
### YOUR CODE HERE ###

# Asignar a la tabla de resultados para mostrar la comparación
results = rf_val_scores
results


# Notice that the scores went down from the training scores across all metrics, but only by very little. This means that the model did not overfit the training data.

# #### **XGBoost**
# 
# Now, do the same thing to get the performance scores of the XGBoost model on the validation data.

# In[39]:


# 1. Use XGBoost model to predict on validation data
# Generar predicciones binarias usando el mejor modelo entrenado de XGBoost
xgb_val_preds = xgb_cv.predict(X_val)

# 2. Get validation scores for XGBoost model
# Calcular las métricas de rendimiento en el conjunto de validación independiente
xgb_val_scores = get_test_scores('XGBoost Val', xgb_val_preds, y_val)

# 3. Append to the results table
# Concatenar la fila de XGBoost debajo de la fila existente de Random Forest
results = pd.concat([results, xgb_val_scores], axis=0).reset_index(drop=True)
results


# Just like with the random forest model, the XGBoost model's validation scores were lower, but only very slightly. It is still the clear champion.

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 12. Use champion model to predict on test data**
# 
# Now, use the champion model to predict on the test dataset. This is to give a final indication of how you should expect the model to perform on new future data, should you decide to use the model.

# In[40]:


# 1. Use XGBoost model to predict on test data
# Generar las predicciones finales usando el modelo campeón de XGBoost sobre el set de prueba
xgb_test_preds = xgb_cv.predict(X_test)

# 2. Get test scores for XGBoost model
# Calcular las métricas finales de producción usando la función get_test_scores
xgb_test_scores = get_test_scores('XGBoost Test', xgb_test_preds, y_test)

# 3. Append to the results table
# Concatenar el rendimiento final de prueba debajo de los resultados anteriores
results = pd.concat([results, xgb_test_scores], axis=0).reset_index(drop=True)
results


# The recall was exactly the same as it was on the validation data, but the precision declined notably, which caused all of the other scores to drop slightly. Nonetheless, this is stil within the acceptable range for performance discrepancy between validation and test scores.

# ### **Task 13. Confusion matrix**
# 
# Plot a confusion matrix of the champion model's predictions on the test data.

# In[41]:


# 1. Generate array of values for confusion matrix
# Generar el arreglo matemático de valores para la matriz de confusión
cm = confusion_matrix(y_test, xgb_test_preds, labels=xgb_cv.classes_)

# 2. Plot confusion matrix using ConfusionMatrixDisplay
# Graficar la matriz de confusión de manera visual y profesional
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
disp.plot(cmap='Blues', values_format='d')


# The model predicted three times as many false negatives than it did false positives, and it correctly identified only 16.6% of the users who actually churned.

# ### **Task 14. Feature importance**
# 
# Use the `plot_importance` function to inspect the most important features of your final model.

# In[42]:


### YOUR CODE HERE ###
# Graficar la importancia de las variables del modelo campeón de XGBoost
fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(xgb_cv.best_estimator_, ax=ax, max_num_features=15, importance_type='weight')
plt.show()


# The XGBoost model made more use of many of the features than did the logistic regression model from the previous course, which weighted a single feature (`activity_days`) very heavily in its final prediction.
# 
# If anything, this underscores the importance of feature engineering. Notice that engineered features accounted for six of the top 10 features (and three of the top five). Feature engineering is often one of the best and easiest ways to boost model performance.
# 
# Also, note that the important features in one model might not be the same as the important features in another model. That's why you shouldn't discount features as unimportant without thoroughly examining them and understanding their relationship with the dependent variable, if possible. These discrepancies between features selected by models are typically caused by complex feature interactions.
# 
# Remember, sometimes your data simply will not be predictive of your chosen target. This is common. Machine learning is a powerful tool, but it is not magic. If your data does not contain predictive signal, even the most complex algorithm will not be able to deliver consistent and accurate predictions. Do not be afraid to draw this conclusion.
# 
# Even if you cannot use the model to make strong predictions, was the work done in vain? What insights can you report back to stakeholders?
# 
# EN: Strategic Value & Stakeholder Insights
# The Work Was Not in Vain: A predictive model does not need to achieve a perfect 90% score to deliver substantial business value. Instead of an automated execution tool, the model serves as an advanced diagnostic asset that maps complex user dynamics.
# 
# Key Reporting Insights for Stakeholders:
# 
# Behavioral Root Causes: We discovered that user churn in Waze is heavily dictated by dynamic speed patterns (km_per_hour) and immediate tenure drop-offs (n_days_after_onboarding), rather than flat historical engagement volume like driving_days.
# 
# High-ROI Target Groups: The confusion matrix proves we can successfully isolate a core group of at-risk users (84 verified churners). This allows the marketing team to deploy highly optimized campaigns, minimizing budget waste on loyal drivers.
# 
# Data Architecture Recommendations: The engineering team now has a clear roadmap. To strengthen future model iterations, Waze needs to engineer features that capture granular, weekly sentiment or micro-drops in driving frequency rather than broad aggregated metrics.
# 
# ES: Valor Estratégico e Insights para Stakeholders
# El Trabajo No Fue en Vano: Un modelo predictivo no necesita alcanzar una puntuación perfecta del 90% para entregar un valor de negocio sustancial. En lugar de una herramienta de ejecución automatizada, el modelo actúa como un activo de diagnóstico avanzado que mapea dinámicas complejas de los usuarios.
# 
# Insights Clave para Reportar a los Directivos:
# 
# Raíces del Comportamiento: Descubrimos que el abandono de usuarios en Waze está fuertemente dictado por patrones dinámicos de velocidad (km_per_hour) y caídas inmediatas en la antigüedad tras el registro (n_days_after_onboarding), en lugar de volúmenes históricos planos como driving_days.
# 
# Grupos Objetivo de Alto ROI: La matriz de confusión demuestra que podemos aislar con éxito un grupo central de usuarios en riesgo (84 abandonos verificados). Esto permite al equipo de marketing desplegar campañas altamente optimizadas, minimizando el desperdicio de presupuesto en conductores leales.
# 
# Recomendaciones de Arquitectura de Datos: El equipo de ingeniería ahora tiene una hoja de ruta clara. Para fortalecer futuras iteraciones del modelo, Waze necesita construir variables que capturen sentimientos semanales granulares o micro-caídas en la frecuencia de conducción, en lugar de métricas agregadas tan amplias.

# ### **Task 15. Conclusion**
# 
# Now that you've built and tested your machine learning models, the next step is to share your findings with the Waze leadership team. Consider the following questions as you prepare to write your executive summary. Think about key points you may want to share with the team, and what information is most relevant to the user churn project.
# 
# **Questions:**
# 
# 1. Would you recommend using this model for churn prediction? Why or why not?
# 
# 2. What tradeoff was made by splitting the data into training, validation, and test sets as opposed to just training and test sets?
# 
# 3. What is the benefit of using a logistic regression model over an ensemble of tree-based models (like random forest or XGBoost) for classification tasks?
# 
# 4. What is the benefit of using an ensemble of tree-based models like random forest or XGBoost over a logistic regression model for classification tasks?
# 
# 5. What could you do to improve this model?
# 
# 6. What additional features would you like to have to help improve the model?
# 
# 

# Q1: Recomendación del Modelo / Model Recommendation
# EN: It depends on the operational goal. I would recommend it for pilot testing, but not for fully automated mitigation campaigns. With a Recall of 16.57%, the model captures roughly 1 out of every 6 real churners. While this is a massive upgrade over Random Forest (5.52%), it leaves a large blind spot. However, because the Precision is 38.89%, the risk of generating unnecessary costs via false alarms is low, making it safe for non-expensive retention actions (like customized in-app push notifications).
# 
# ES: Depende del objetivo operativo. Lo recomendaría para pruebas piloto, pero no para campañas de mitigación completamente automatizadas. Con un Recall del 16.57%, el modelo captura aproximadamente 1 de cada 6 usuarios en abandono real. Aunque es una mejora masiva sobre Random Forest (5.52%), deja un punto ciego grande. Sin embargo, debido a que la Precisión es del 38.89%, el riesgo de generar costos innecesarios por falsas alarmas es bajo, lo que lo hace seguro para acciones de retención de bajo costo (como notificaciones push personalizadas dentro de la app).
# 
# Q2: Compromiso de División de Datos / Data Splitting Trade-off (3-Way Split)
# EN: The core trade-off is validation purity versus training volume. By keeping a 3-way split (Train, Validation, Test), we reduce the total amount of rows available for the model to learn raw patterns during the .fit() stage. However, the advantage is monumental: it prevents hyperparameter leakage. The validation set allows us to compare and tune Random Forest against XGBoost freely, preserving the Test set as a truly unpolluted benchmark to estimate real production performance.
# 
# ES: El compromiso central es la pureza de la validación frente al volumen de entrenamiento. Al mantener una división de tres vías (Entrenamiento, Validación, Prueba), reducimos la cantidad total de filas disponibles para que el modelo aprenda patrones puros en la etapa .fit(). Sin embargo, la ventaja es monumental: evita la fuga en la optimización. El conjunto de validación nos permite comparar y ajustar Random Forest contra XGBoost libremente, preservando el conjunto de Prueba como un referente realmente limpio para estimar el rendimiento real en producción.
# 
# Q3: Ventaja de la Regresión Logística / Advantage of Logistic Regression
# EN: Extreme interpretability and computational efficiency. A Logistic Regression computes in fractions of a second and outputs direct odds-ratios (coefficients). This allows stakeholders to clearly understand the linear direction and weight of every single feature (e.g., exactly how much the probability of churn increases for every additional kilometer driven).
# 
# ES: Extreme interpretability and computational efficiency. Una Regresión Logística calcula en fracciones de segundo y entrega razones de momios (coeficientes) directas. Esto permite a los stakeholders entender con total claridad la dirección lineal y el peso de cada variable (por ejemplo, exactamente cuánto aumenta la probabilidad de abandono por cada kilómetro adicional conducido).
# 
# Q4: Ventaja de los Ensambles de Árboles / Advantage of Tree Ensembles
# EN: Capture of complex non-linear boundaries and high resilience to imbalance. Tree-based models don't assume data behaves as a straight line. Algorithms like XGBoost construct sequential rules that easily capture complex interaction thresholds (e.g., user is only at risk if speed is high and app days are low), resulting in much higher predictive power (Recall scaled from Logistic Regression baselines).
# 
# ES: Captura de límites complejos no lineales y alta resiliencia al desbalance. Los modelos basados en árboles no asumen que los datos se comportan como una línea recta. Algoritmos como XGBoost conruyen reglas secuenciales que capturan fácilmente umbrales de interacción complejos (por ejemplo, el usuario solo está en riesgo si la velocidad es alta y los días en la app son bajos), resultando en un poder predictivo muy superior.
# 
# Q5: Cómo mejorar este modelo / How to Improve the Current Model
# EN:
# 
# Feature Engineering: Create granular ratio features, such as the acceleration drop rate or weekly consistency metrics instead of monthly aggregates.
# 
# Anomalous Threshold Tuning: Adjust the default classification threshold (moving it down from 0.5 to 0.3 or 0.4) to aggressively force higher Recall, capturing more churners at the cost of some extra false alarms.
# 
# ES:
# 
# Ingeniería de Variables: Crear variables de tasas granulares, como la tasa de caída en la aceleración o métricas de consistencia semanales en lugar de agregados mensuales.
# 
# Ajuste de Umbral Anómalo: Ajustar el umbral de clasificación por defecto (bajándolo de 0.5 a 0.3 o 0.4) para forzar un Recall más agresivo, capturando más usuarios en riesgo a costa de algunas falsas alarmas extra.
# 
# Q6: Características Adicionales Deseadas / Ideal Additional Features
# EN:
# 
# Granular Temporal Logs: Changes in weekly drive frequency (to pinpoint the exact moment usage starts to decay).
# 
# App Feedback Interaction: User reports on UI bugs, connection drops, or negative feedback inside the app.
# 
# Destination Type Data: Commute behavior (e.g., whether they use Waze for fixed work trips or sporadic leisure trips).
# 
# ES:
# 
# Registros Temporales Granulares: Cambios en la frecuencia de conducción semanal (para identificar el momento exacto en que decae el uso).
# 
# Interacción con Errores de la App: Reportes del usuario sobre fallos de interfaz, caídas de conexión o comentarios negativos dentro de la aplicación.
# 
# Datos del Tipo de Destino: Comportamiento de viaje diario (por ejemplo, si usan Waze para viajes fijos al trabajo o viajes esporádicos de ocio).

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
