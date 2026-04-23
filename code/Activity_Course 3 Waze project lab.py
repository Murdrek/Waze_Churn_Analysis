#!/usr/bin/env python
# coding: utf-8

# # **Waze Project**
# **Course 2 - Go Beyond the Numbers: Translate Data into Insights**

# Your team is still in the early stages of their user churn project. So far, you’ve completed a project proposal and used Python to inspect and organize Waze’s user data.
# 
# You check your inbox and notice a new message from Chidi Ga, your team’s Senior Data Analyst. Chidi is pleased with the work you have already completed and requests your assistance with exploratory data analysis (EDA) and further data visualization. Harriet Hadzic, Waze's Director of Data Analysis, will want to review a Python notebook that shows your data exploration and visualization.
# 
# A notebook was structured and prepared to help you in this project. Please complete the following questions and prepare an executive summary.

# # **Course 2 End-of-course project: Exploratory data analysis**
# 
# In this activity, you will examine data provided and prepare it for analysis.
# <br/>
# 
# **The purpose** of this project is to conduct exploratory data analysis (EDA) on a provided dataset.
# 
# **The goal** is to continue the examination of the data that you began in the previous Course, adding relevant visualizations that help communicate the story that the data tells.
# <br/>
# 
# 
# *This activity has 4 parts:*
# 
# **Part 1:** Imports, links, and loading
# 
# **Part 2:** Data Exploration
# *   Data cleaning
# 
# 
# **Part 3:** Building visualizations
# 
# **Part 4:** Evaluating and sharing results
# 
# <br/>
# 
# 
# Follow the instructions and answer the question below to complete the activity. Then, you will complete an executive summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.

# # **Visualize a story in Python**

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
# 

# ### **Task 1. Imports and data loading**
# 
# For EDA of the data, import the data and packages that will be most helpful, such as pandas, numpy, and matplotlib.
# 
# 
# 

# In[56]:


### YOUR CODE HERE ###
# Task 1. Imports and data loading / Importación y carga de datos

# EN: Import packages for data manipulation and visualization
# ES: Importar paquetes para manipulación y visualización de datos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Read in the data and store it as a dataframe object called df.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[57]:


# Load the dataset into a dataframe
df = pd.read_csv('waze_dataset.csv')


# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document and those below where applicable to complete your code:
# 1. Does the data need to be restructured or converted into usable formats?
# 
# 2. Are there any variables that have missing data?
# 

# ==> ENTER YOUR RESPONSES TO QUESTIONS 1-2 HERE
# EN: Does the data need to be restructured or converted into usable formats?
# ES: ¿Es necesario reestructurar los datos o convertirlos a formatos utilizables?
# 
# EN: The data is mostly well-structured. However, the label column (target) is an object and may need conversion to binary (0/1) for future modeling. Also, checking for inconsistencies in driving days vs. activity days is required.
# 
# ES: Los datos están mayoritariamente bien estructurados. Sin embargo, la columna label (objetivo) es un objeto y podría requerir conversión a binario (0/1) para el modelado futuro. También es necesario verificar inconsistencias entre los días de conducción y los días de actividad.
# 
# EN: Are there any variables that have missing data?
# ES: ¿Hay alguna variable que tenga datos faltantes?
# 
# EN: Yes, preliminary inspection shows that the label column has missing values. Since this is our target variable, we must analyze if these rows should be dropped or if there is a pattern in the missingness.
# 
# ES: Sí, la inspección preliminar muestra que la columna label tiene valores faltantes. Dado que esta es nuestra variable objetivo, debemos analizar si estas filas deben eliminarse o si hay un patrón en la falta de datos.

# ### **Task 2. Data exploration and cleaning**
# 
# Consider the following questions:
# 
# 
# 
# 1.  Given the scenario, which data columns are most applicable?
# 
# 2.  Which data columns can you eliminate, knowing they won’t solve your problem scenario?
# 
# 3.  How would you check for missing data? And how would you handle missing data (if any)?
# 
# 4.  How would you check for outliers? And how would handle outliers (if any)?
# 
# 
# 
# 
# 
# 

# ==> ENTER YOUR RESPONSES TO QUESTIONS 1-4 HERE
# 
# EN: Which data columns are most applicable?
# ES: ¿Qué columnas de datos son las más aplicables?
# 
# EN: The most critical columns are label (target), activity_days, driving_days, sessions, and km_driven. These directly measure user engagement and the value they derive from the app. n_days_after_onboarding is also vital to determine if churn is related to user experience duration.
# 
# ES: Las columnas más críticas son label (objetivo), activity_days, driving_days, sessions y km_driven. Estas miden directamente el compromiso del usuario y el valor que obtienen de la app. n_days_after_onboarding también es vital para determinar si la rotación está relacionada con la antigüedad del usuario.
# 
# 2. Elimination / Eliminación
# EN: Which data columns can you eliminate?
# ES: ¿Qué columnas de datos puedes eliminar?
# 
# EN: For an initial predictive model, total_navigations_fav1 and total_navigations_fav2 might be less relevant than general usage metrics, as they represent specific habits rather than overall churn trends. However, I would keep them initially during EDA to check for correlations before dropping them.
# 
# ES: Para un modelo predictivo inicial, total_navigations_fav1 y total_navigations_fav2 podrían ser menos relevantes que las métricas de uso general, ya que representan hábitos específicos en lugar de tendencias generales de rotación. Sin embargo, las mantendría inicialmente durante el EDA para verificar correlaciones antes de descartarlas.
# 
# 3. Missing Data / Datos Faltantes
# EN: How would you check for and handle missing data?
# ES: ¿Cómo verificarías y manejarías los datos faltantes?
# 
# EN: I would check using df.isnull().sum(). Since the missing values are in the label column (~4.7%), I would drop these rows. Imputing the target variable could introduce significant bias into the model's predictions.
# 
# ES: Verificaría usando df.isnull().sum(). Dado que los valores faltantes están en la columna label (~4.7%), eliminaría estas filas. Imputar la variable objetivo podría introducir un sesgo significativo en las predicciones del modelo.
# 
# 4. Outliers / Valores Atípicos
# EN: How would you check for and handle outliers?
# ES: ¿Cómo verificarías y manejarías los valores atípicos?
# 
# EN: I would use boxplots and the df.describe() method to identify values beyond the Interquartile Range (IQR). For handling, I would consider "capping" or "winsorizing" the data (setting outliers to a maximum threshold like the 95th percentile) to prevent extreme values from distorting the model's logic.
# 
# ES: Usaría boxplots (diagramas de caja) y el método df.describe() para identificar valores más allá del Rango Intercuartílico (IQR). Para manejarlos, consideraría el "capping" o "winsorizing" (ajustar los valores atípicos a un umbral máximo como el percentil 95) para evitar que valores extremos distorsionen la lógica del modelo.

# #### **Data overview and summary statistics**
# 
# Use the following methods and attributes on the dataframe:
# 
# * `head()`
# * `size`
# * `describe()`
# * `info()`
# 
# It's always helpful to have this information at the beginning of a project, where you can always refer back to if needed.

# In[58]:


### YOUR CODE HERE ###
# --- Data Overview and Summary Statistics / Descripción General y Estadísticas ---

# 1. EN: Display the first 5 rows to see the data structure
#    ES: Mostrar las primeras 5 filas para ver la estructura de los datos
print("--- head() ---")
display(df.head())


# In[59]:


### YOUR CODE HERE ###

# 2. EN: Check the total number of elements in the dataframe (rows * columns)
#    ES: Verificar el número total de elementos en el dataframe (filas * columnas)
print(f"--- size ---\nTotal elements / Elementos totales: {df.size}\n")


# Generate summary statistics using the `describe()` method.

# In[60]:


### YOUR CODE HERE ###
# 3. EN: Generate summary statistics to identify distribution and potential outliers
#    ES: Generar estadísticas descriptivas para identificar la distribución y posibles atípicos
print("--- describe() ---")
display(df.describe())


# And summary information using the `info()` method.

# In[61]:


### YOUR CODE HERE ###
# 4. EN: Review data types and count non-null values
#    ES: Revisar tipos de datos y contar valores no nulos
print("--- info() ---")
df.info()


# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Construct stage.

# Consider the following questions as you prepare to deal with outliers:
# 
# 1.   What are some ways to identify outliers?
# 2.   How do you make the decision to keep or exclude outliers from any future models?

# ==> ENTER YOUR RESPONSES TO QUESTIONS 1-2 HERE
# EN: What are some ways to identify outliers?
# ES: ¿Cuáles son algunas formas de identificar valores atípicos?
# 
# EN: * Boxplots: Visually identify points beyond the whiskers (usually $1.5 \times$ IQR).
# 
# Z-Score: Values that are more than 3 standard deviations from the mean.
# 
# Interquartile Range (IQR): Mathematically calculating thresholds ($Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$).
# 
# Descriptive Statistics: Using df.describe() to spot massive gaps between the 75th percentile and the Max value.
# 
# ES: * Boxplots (Diagramas de caja): Identificar visualmente puntos más allá de los "bigotes" (usualmente $1.5 \times$ el IQR).
# 
# Z-Score: Valores que están a más de 3 desviaciones estándar de la media.Rango Intercuartílico (IQR): Calcular matemáticamente los umbrales superior e inferior.
# 
# Estadística Descriptiva: Usar df.describe() para detectar brechas masivas entre el percentil 75 y el valor máximo.
# 
# EN: How do you make the decision to keep or exclude outliers?
# ES: ¿Cómo se toma la decisión de mantener o excluir los valores atípicos?
# 
# Keep them: If they represent legitimate user behavior (e.g., a real power-user who drives 8 hours a day).
# 
# Delete them: If they are clearly data entry errors or sensor glitches (e.g., driving 24 hours non-stop).
# 
# Capping/Winsorizing: If the data is real but extreme enough to distort the model's coefficients. We replace extreme values with the 95th or 99th percentile.
# 
# ES:
# 
# Mantenerlos: Si representan un comportamiento de usuario legítimo (ej: un "power-user" real que conduce 8 horas diarias).
# 
# Eliminarlos: Si son claramente errores de entrada de datos o fallos del sensor (ej: conducir 24 horas sin parar).
# 
# Capping (Limitación): Si los datos son reales pero lo suficientemente extremos como para distorsionar los coeficientes del modelo. Reemplazamos los valores extremos con el percentil 95 o 99.
# 
# 

# ### **Task 3a. Visualizations**
# 
# Select data visualization types that will help you understand and explain the data.
# 
# Now that you know which data columns you’ll use, it is time to decide which data visualization makes the most sense for EDA of the Waze dataset.
# 
# **Question:** What type of data visualization(s) will be most helpful?
# 
# * Line graph
# * Bar chart
# * Box plot
# * Histogram
# * Heat map
# * Scatter plot
# * A geographic map
# 
# 

# ==> ENTER YOUR RESPONSE HERE
# 
# EN: What type of data visualization(s) will be most helpful?
# ES: ¿Qué tipo de visualización(es) de datos serán más útiles?
# 
# Box plots:
# 
# EN: Essential for identifying outliers in driving distance and session counts.
# ES: Esenciales para identificar valores atípicos en la distancia recorrida y el conteo de sesiones.
# 
# Histograms:
# 
# EN: Necessary to understand the distribution of variables like sessions and driving_days (e.g., normal distribution vs. skewed).
# ES: Necesarios para entender la distribución de variables como sessions y driving_days (ej. distribución normal vs. sesgada).
# 
# Bar charts:
# 
# EN: Ideal for comparing churn vs. retained users across different categories like device.
# ES: Ideales para comparar usuarios que se dan de baja vs. retenidos en diferentes categorías como device.
# 
# Scatter plots:
# 
# EN: Useful for exploring the relationship between two continuous variables, such as sessions and km_driven.
# ES: Útiles para explorar la relación entre dos variables continuas, como sessions y km_driven.

# Begin by examining the spread and distribution of important variables using box plots and histograms.

# #### **`sessions`**
# 
# _The number of occurrence of a user opening the app during the month_

# In[62]:


# Box plot
### YOUR CODE HERE ###
# Task 3. Visualizations - sessions
# Tarea 3. Visualizaciones - sesiones

# EN: 1. Create a box plot to examine the spread and identify outliers
# ES: 1. Crear un diagrama de caja para examinar la dispersión e identificar atípicos
plt.figure(figsize=(10, 2))
sns.boxplot(x=df['sessions'], fliersize=3)

# EN: Add title and labels
# ES: Agregar título y etiquetas
plt.title('Sessions Box Plot / Diagrama de Caja de Sesiones')
plt.xlabel('Number of sessions / Número de sesiones')

plt.show()


# In[63]:


# Histogram
### YOUR CODE HERE ###
# Task 3. Visualizations - sessions (Histogram)
# Tarea 3. Visualizaciones - sesiones (Histograma)

# EN: Create a histogram to visualize the frequency distribution of sessions
# ES: Crear un histograma para visualizar la distribución de frecuencia de las sesiones
plt.figure(figsize=(10, 5))
sns.histplot(df['sessions'], bins=range(0, 750, 25), kde=True)

# EN: Add titles and axis labels
# ES: Agregar títulos y etiquetas de los ejes
plt.title('Sessions Histogram / Histograma de Sesiones')
plt.xlabel('Number of sessions / Número de sesiones')
plt.ylabel('Frequency / Frecuencia')

plt.show()


# The `sessions` variable is a right-skewed distribution with half of the observations having 56 or fewer sessions. However, as indicated by the boxplot, some users have more than 700.

# #### **`drives`**
# 
# _An occurrence of driving at least 1 km during the month_

# In[64]:


# Box plot
### YOUR CODE HERE ###
# EN: 1. Box plot to identify outliers in driving frequency
# ES: 1. Box plot para identificar atípicos en la frecuencia de viajes
plt.figure(figsize=(10, 2))
sns.boxplot(x=df['drives'], fliersize=3)
plt.title('Drives Box Plot / Diagrama de Caja de Viajes')
plt.xlabel('Number of drives / Número de viajes')
plt.show()


# In[38]:


# Histogram
### YOUR CODE HERE ###
# EN: 2. Histogram to visualize the distribution of drives
# ES: 2. Histograma para visualizar la distribución de los viajes
plt.figure(figsize=(10, 5))
sns.histplot(df['drives'], bins=range(0, 600, 20), kde=True)
plt.title('Drives Histogram / Histograma de Viajes')
plt.xlabel('Number of drives / Número de viajes')
plt.ylabel('Frequency / Frecuencia')
plt.show()


# The `drives` information follows a distribution similar to the `sessions` variable. It is right-skewed, approximately log-normal, with a median of 48. However, some drivers had over 400 drives in the last month.

# #### **`total_sessions`**
# 
# _A model estimate of the total number of sessions since a user has onboarded_

# In[65]:


# Box plot
### YOUR CODE HERE ###
# 1. Box plot
plt.figure(figsize=(10, 2))
sns.boxplot(x=df['total_sessions'], fliersize=3)
plt.title('Total Sessions Box Plot / Acumulado Histórico de Sesiones')
plt.xlabel('Number of total sessions / Número total de sesiones')
plt.show()


# In[66]:


# Histogram
### YOUR CODE HERE ###
# 2. Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df['total_sessions'], bins=range(0, 1600, 50), kde=True)
plt.title('Total Sessions Histogram / Histograma de Sesiones Totales')
plt.xlabel('Number of total sessions / Número total de sesiones')
plt.ylabel('Frequency / Frecuencia')
plt.show()


# The `total_sessions` is a right-skewed distribution. The median total number of sessions is 159.6. This is interesting information because, if the median number of sessions in the last month was 48 and the median total sessions was ~160, then it seems that a large proportion of a user's total drives might have taken place in the last month. This is something you can examine more closely later.

# #### **`n_days_after_onboarding`**
# 
# _The number of days since a user signed up for the app_

# In[67]:


# Box plot
### YOUR CODE HERE ###
# Task 3. Visualizations - n_days_after_onboarding
# Análisis de la antigüedad de los usuarios en días

plt.figure(figsize=(10, 2))
sns.boxplot(x=df['n_days_after_onboarding'], fliersize=3)
plt.title('Onboarding Tenure Box Plot / Antigüedad desde Onboarding')
plt.xlabel('Days after onboarding / Días desde el registro')
plt.show()


# In[68]:


# Histogram
### YOUR CODE HERE ###
# 2. Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df['n_days_after_onboarding'], bins=range(0, 3600, 100))
plt.title('Onboarding Tenure Histogram / Histograma de Antigüedad')
plt.xlabel('Days after onboarding / Días desde el registro')
plt.ylabel('Frequency / Frecuencia')
plt.show()


# The total user tenure (i.e., number of days since
# onboarding) is a uniform distribution with values ranging from near-zero to \~3,500 (\~9.5 years).

# #### **`driven_km_drives`**
# 
# _Total kilometers driven during the month_

# In[69]:


# Box plot
### YOUR CODE HERE ###
# Task 3. Visualizations - driven_km_drives
# Análisis de la distancia recorrida en el mes

# 1. Box plot
plt.figure(figsize=(10, 2))
sns.boxplot(x=df['driven_km_drives'], fliersize=3)
plt.title('Distance Box Plot / Diagrama de Caja de Distancia (km)')
plt.xlabel('Kilometers driven / Kilómetros recorridos')
plt.show()


# In[70]:


# Histogram
### YOUR CODE HERE ###
# Histogram
### YOUR CODE HERE ###
# 2. Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df['driven_km_drives'], bins=range(0, 22000, 500), kde=True)
plt.title('Distance Histogram / Histograma de Distancia')
plt.xlabel('Kilometers driven / Kilómetros recorridos')
plt.ylabel('Frequency / Frecuencia')
plt.show()


# The number of drives driven in the last month per user is a right-skewed distribution with half the users driving under 3,495 kilometers. As you discovered in the analysis from the previous course, the users in this dataset drive _a lot_. The longest distance driven in the month was over half the circumferene of the earth.

# #### **`duration_minutes_drives`**
# 
# _Total duration driven in minutes during the month_

# In[71]:


# Box plot
### YOUR CODE HERE ###
# Task 3. Visualizations - duration_minutes_drives
# Análisis del tiempo total de conducción en el mes

# 1. Box plot
plt.figure(figsize=(10, 2))
sns.boxplot(x=df['duration_minutes_drives'], fliersize=3)
plt.title('Duration Box Plot / Diagrama de Caja de Duración (min)')
plt.xlabel('Duration in minutes / Duración en minutos')
plt.show()


# In[73]:


# Histogram
### YOUR CODE HERE ###
# 2. Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df['duration_minutes_drives'], bins=range(0, 16000, 500), kde=True)
plt.title('Duration Histogram / Histograma de Duración')
plt.xlabel('Duration in minutes / Duración en minutos')
plt.ylabel('Frequency / Frecuencia')
plt.show()


# The `duration_minutes_drives` variable has a heavily skewed right tail. Half of the users drove less than \~1,478 minutes (\~25 hours), but some users clocked over 250 hours over the month.

# #### **`activity_days`**
# 
# _Number of days the user opens the app during the month_

# In[74]:


# Box plot
### YOUR CODE HERE ###
# Task 3. Visualizations - activity_days
# Análisis de la frecuencia de uso mensual

# 1. Box plot
plt.figure(figsize=(10, 2))
sns.boxplot(x=df['activity_days'], fliersize=3)
plt.title('Activity Days Box Plot / Días de Actividad (Mensual)')
plt.xlabel('Days / Días')
plt.show()


# In[45]:


# Histogram
### YOUR CODE HERE ###
# 2. Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df['activity_days'], bins=range(0, 32, 1))
plt.title('Activity Days Histogram / Histograma de Días de Actividad')
plt.xlabel('Days / Días')
plt.ylabel('Frequency / Frecuencia')
plt.show()


# Within the last month, users opened the app a median of 16 times. The box plot reveals a centered distribution. The histogram shows a nearly uniform distribution of ~500 people opening the app on each count of days. However, there are ~250 people who didn't open the app at all and ~250 people who opened the app every day of the month.
# 
# This distribution is noteworthy because it does not mirror the `sessions` distribution, which you might think would be closely correlated with `activity_days`.

# #### **`driving_days`**
# 
# _Number of days the user drives (at least 1 km) during the month_

# In[75]:


# Box plot
### YOUR CODE HERE ###
# Task 3. Visualizations - driving_days
# Análisis de los días de conducción efectiva
plt.figure(figsize=(10, 2))
sns.boxplot(x=df['driving_days'], fliersize=3)
plt.title('Driving Days Box Plot / Días de Conducción (Mensual)')
plt.xlabel('Days / Días')
plt.show()


# In[47]:


# Histogram
### YOUR CODE HERE ###
# 2. Histogram
plt.figure(figsize=(10, 5))
sns.histplot(df['driving_days'], bins=range(0, 32, 1))
plt.title('Driving Days Histogram / Histograma de Días de Conducción')
plt.xlabel('Days / Días')
plt.ylabel('Frequency / Frecuencia')
plt.show()


# The number of days users drove each month is almost uniform, and it largely correlates with the number of days they opened the app that month, except the `driving_days` distribution tails off on the right.
# 
# However, there were almost twice as many users (\~1,000 vs. \~550) who did not drive at all during the month. This might seem counterintuitive when considered together with the information from `activity_days`. That variable had \~500 users opening the app on each of most of the day counts, but there were only \~250 users who did not open the app at all during the month and ~250 users who opened the app every day. Flag this for further investigation later.

# #### **`device`**
# 
# _The type of device a user starts a session with_
# 
# This is a categorical variable, so you do not plot a box plot for it. A good plot for a binary categorical variable is a pie chart.

# In[76]:


# Pie chart
### YOUR CODE HERE ###
# Task 3. Visualizations - device
# Análisis de la distribución por tipo de dispositivo

# 1. Preparar los datos
device_counts = df['device'].value_counts()

# 2. Pie chart
plt.figure(figsize=(7, 7))
plt.pie(device_counts, 
        labels=device_counts.index, 
        autopct='%1.1f%%', 
        colors=['#4c8bf5', '#34a853'], 
        startangle=140,
        explode=(0.05, 0)) # Resalta ligeramente la porción mayor
plt.title('User Distribution by Device / Distribución de Usuarios por Dispositivo')
plt.show()


# There are nearly twice as many iPhone users as Android users represented in this data.

# #### **`label`**
# 
# _Binary target variable (“retained” vs “churned”) for if a user has churned anytime during the course of the month_
# 
# This is also a categorical variable, and as such would not be plotted as a box plot. Plot a pie chart instead.

# In[77]:


# Pie chart
### YOUR CODE HERE ###
# Task 3. Visualizations - label
# Análisis de la variable objetivo: ¿Cuántos usuarios perdemos?

# 1. Preparar los datos
label_counts = df['label'].value_counts()

# 2. Pie chart
plt.figure(figsize=(7, 7))
plt.pie(label_counts, 
        labels=label_counts.index, 
        autopct='%1.1f%%', 
        colors=['#24cdff', '#ff5a5f'], # Azul para retenidos, Rojo para fuga
        startangle=140,
        explode=(0.1, 0)) # Resaltamos la fuga
plt.title('Retention vs. Churn / Retención vs. Fuga')
plt.show()


# Less than 18% of the users churned.

# #### **`driving_days` vs. `activity_days`**
# 
# Because both `driving_days` and `activity_days` represent counts of days over a month and they're also closely related, you can plot them together on a single histogram. This will help to better understand how they relate to each other without having to scroll back and forth comparing histograms in two different places.
# 
# Plot a histogram that, for each day, has a bar representing the counts of `driving_days` and `activity_days`.

# In[78]:


# Histogram
### YOUR CODE HERE ###
# Task 3. Visualizations - Comparison
# Comparación de Días de Actividad vs. Días de Conducción

plt.figure(figsize=(12, 5))
label = ['driving_days', 'activity_days']
plt.hist([df['driving_days'], df['activity_days']],
         bins=range(0, 33),
         label=label,
         color=['#34a853', '#4c8bf5']) # Verde para conducir, Azul para actividad

plt.title('Driving Days vs. Activity Days / Días de Conducción vs. Actividad')
plt.xlabel('Days / Días')
plt.ylabel('User Count / Conteo de Usuarios')
plt.legend()
plt.show()


# As observed previously, this might seem counterintuitive. After all, why are there _fewer_ people who didn't use the app at all during the month and _more_ people who didn't drive at all during the month?
# 
# On the other hand, it could just be illustrative of the fact that, while these variables are related to each other, they're not the same. People probably just open the app more than they use the app to drive&mdash;perhaps to check drive times or route information, to update settings, or even just by mistake.
# 
# Nonetheless, it might be worthwile to contact the data team at Waze to get more information about this, especially because it seems that the number of days in the month is not the same between variables.
# 
# Confirm the maximum number of days for each variable&mdash;`driving_days` and `activity_days`.

# In[79]:


### YOUR CODE HERE ###
### YOUR CODE HERE ###
# Validating the maximum and minimum values for both day-count variables
# Validando los valores máximos y mínimos para ambas variables de conteo de días

validation_metrics = df[['driving_days', 'activity_days']].agg(['min', 'max', 'mean', 'median'])
print(validation_metrics)

# Checking if there are records where driving_days > activity_days
# Verificando si existen registros donde los días de conducción superan los de actividad
inconsistency_check = df[df['driving_days'] > df['activity_days']].shape[0]
print(f"\nTotal inconsistent rows (Driving > Activity): {inconsistency_check}")


# It's true. Although it's possible that not a single user drove all 31 days of the month, it's highly unlikely, considering there are 15,000 people represented in the dataset.
# 
# One other way to check the validity of these variables is to plot a simple scatter plot with the x-axis representing one variable and the y-axis representing the other.

# In[80]:


# Scatter plot
### YOUR CODE HERE ###
# Scatter plot
### YOUR CODE HERE ###
# Task 3. Visualizations - Scatter Plot
# Relación entre Días de Actividad y Días de Conducción

plt.figure(figsize=(8, 8))
sns.scatterplot(data=df, x='activity_days', y='driving_days', alpha=0.3)

# Añadimos una línea de referencia y = x (donde actividad = conducción)
plt.plot([0, 31], [0, 31], color='red', linestyle='--')

plt.title('Activity Days vs. Driving Days Scatter Plot')
plt.xlabel('Activity Days')
plt.ylabel('Driving Days')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


# Notice that there is a theoretical limit. If you use the app to drive, then by definition it must count as a day-use as well. In other words, you cannot have more drive-days than activity-days. None of the samples in this data violate this rule, which is good.

# #### **Retention by device**
# 
# Plot a histogram that has four bars&mdash;one for each device-label combination&mdash;to show how many iPhone users were retained/churned and how many Android users were retained/churned.

# In[81]:


# Histogram
### YOUR CODE HERE ###
# Task 3. Visualizations - Retention by Device
# ¿El dispositivo afecta la fuga de usuarios?

# Gráfico de barras agrupadas con valores numéricos
plt.figure(figsize=(10, 6))

# Creamos el gráfico y lo asignamos a una variable 'ax' para poder editarlo
ax = sns.countplot(data=df, x='device', hue='label', palette='viridis')

# Agregamos los valores sobre cada barra
for container in ax.containers:
    ax.bar_label(container)

plt.title('Retention by Device / Retención por Dispositivo')
plt.xlabel('Device / Dispositivo')
plt.ylabel('User Count / Conteo de Usuarios')
plt.legend(title='Status', labels=['Retained', 'Churned'])

# Ajustamos el límite del eje Y para que los números no se corten
plt.ylim(0, df['device'].value_counts().max() + 1000)

plt.show()


# The proportion of churned users to retained users is consistent between device types.

# #### **Retention by kilometers driven per driving day**
# 
# In the previous course, you discovered that the median distance driven per driving day last month for users who churned was 697.54 km, versus 289.55 km for people who did not churn. Examine this further.
# 
# 1. Create a new column in `df` called `km_per_driving_day`, which represents the mean distance driven per driving day for each user.
# 
# 2. Call the `describe()` method on the new column.

# In[82]:


# 1. Create `km_per_driving_day` column
### YOUR CODE HERE ###
# Creamos la columna dividiendo la distancia total por los días que realmente condujo
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# 2. Call `describe()` on the new column
### YOUR CODE HERE ###
# Obtenemos las estadísticas descriptivas para entender la distribución
km_day_stats = df['km_per_driving_day'].describe()
print(km_day_stats)


# What do you notice? The mean value is infinity, the standard deviation is NaN, and the max value is infinity. Why do you think this is?
# 
# The Cause: Division by Zero / La Causa: División por Cero
# EN: As you noticed, some users in the dataset have 0 in the driving_days column. When creating the km_per_driving_day column, Pandas performs the following operation:
# 
# ES: Como notaste, hay usuarios en el dataset que tienen 0 en la columna driving_days. Al crear la columna km_per_driving_day, Pandas realiza la siguiente operación:
# 
# ![image.png](attachment:a7438f84-e55e-4245-ab77-a4e4d652641f.png)
# 
# EN: In mathematics, division by zero is undefined. Pandas/Numpy handles this by assigning the value inf (infinity).
# 
# ES: En matemáticas, la división por cero es indefinida. Pandas/Numpy maneja esto asignando el valor inf (infinito)
# 
# This is the result of there being values of zero in the `driving_days` column. Pandas imputes a value of infinity in the corresponding rows of the new column because division by zero is undefined.
# 
# 1. Convert these values from infinity to zero. You can use `np.inf` to refer to a value of infinity.
# 
# 2. Call `describe()` on the `km_per_driving_day` column to verify that it worked.

# In[83]:


# 1. Convert infinite values to zero
### YOUR CODE HERE ###

# Reemplazamos np.inf por 0 en la columna específica
df['km_per_driving_day'] = df['km_per_driving_day'].replace(np.inf, 0)

# 2. Confirm that it worked
### YOUR CODE HERE ###
# Al llamar a describe() ahora, verás que mean y std tienen valores numéricos
print(df['km_per_driving_day'].describe())


# The maximum value is 15,420 kilometers _per drive day_. This is physically impossible. Driving 100 km/hour for 12 hours is 1,200 km. It's unlikely many people averaged more than this each day they drove, so, for now, disregard rows where the distance in this column is greater than 1,200 km.
# 
# Plot a histogram of the new `km_per_driving_day` column, disregarding those users with values greater than 1,200 km. Each bar should be the same length and have two colors, one color representing the percent of the users in that bar that churned and the other representing the percent that were retained. This can be done by setting the `multiple` parameter of seaborn's [`histplot()`](https://seaborn.pydata.org/generated/seaborn.histplot.html) function to `fill`.

# In[84]:


# Histogram
### YOUR CODE HERE ###
# 1. Creamos el histograma filtrando los valores > 1200 km
plt.figure(figsize=(12, 5))
sns.histplot(data=df[df['km_per_driving_day'] <= 1200], 
             x='km_per_driving_day', 
             hue='label', 
             multiple='fill', 
             palette='viridis')

# Ajustamos etiquetas para claridad
plt.ylabel('Percent', fontsize=12)
plt.xlabel('km per driving day', fontsize=12)
plt.title('Churn rate by mean km per driving day', fontsize=14)
plt.show()


# The churn rate tends to increase as the mean daily distance driven increases, confirming what was found in the previous course. It would be worth investigating further the reasons for long-distance users to discontinue using the app.

# #### **Churn rate per number of driving days**
# 
# Create another histogram just like the previous one, only this time it should represent the churn rate for each number of driving days.

# In[85]:


# Histogram
### YOUR CODE HERE ###
# 1. Creamos el histograma para la columna 'driving_days'
plt.figure(figsize=(12, 5))
sns.histplot(data=df, 
             x='driving_days', 
             hue='label', 
             multiple='fill', 
             discrete=True, # Usamos discrete=True porque los días son números enteros (0-31)
             palette='viridis')

# Ajustamos etiquetas y título
plt.ylabel('Percent', fontsize=12)
plt.xlabel('driving_days', fontsize=12)
plt.title('Churn rate per number of driving days', fontsize=14)
plt.show()


# The churn rate is highest for people who didn't use Waze much during the last month. The more times they used the app, the less likely they were to churn. While 40% of the users who didn't use the app at all last month churned, nobody who used the app 30 days churned.
# 
# This isn't surprising. If people who used the app a lot churned, it would likely indicate dissatisfaction. When people who don't use the app churn, it might be the result of dissatisfaction in the past, or it might be indicative of a lesser need for a navigational app. Maybe they moved to a city with good public transportation and don't need to drive anymore.

# #### **Proportion of sessions that occurred in the last month**
# 
# Create a new column `percent_sessions_in_last_month` that represents the percentage of each user's total sessions that were logged in their last month of use.

# In[86]:


### YOUR CODE HERE ###
# 1. Creamos la columna calculando el porcentaje
# Fórmula: (sesiones del último mes / sesiones totales)
df['percent_sessions_in_last_month'] = df['sessions'] / df['total_sessions']

# 2. Llamamos a describe() para ver la distribución de esta nueva métrica
print(df['percent_sessions_in_last_month'].describe())


# What is the median value of the new column?

# In[87]:


### YOUR CODE HERE ###
# Calcular el valor mediano de la nueva columna
median_sessions_last_month = df['percent_sessions_in_last_month'].median()

print(f"The median value is: {median_sessions_last_month}")


# Now, create a histogram depicting the distribution of values in this new column.

# In[88]:


# Histogram
### YOUR CODE HERE ###
# Crear el histograma de la nueva columna
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='percent_sessions_in_last_month', hue='label', multiple='stack')

plt.title('Distribution of percent_sessions_in_last_month by Retention Status')
plt.xlabel('Percent of sessions in last month')
plt.ylabel('Number of users')
plt.show()


# Check the median value of the `n_days_after_onboarding` variable.

# In[89]:


### YOUR CODE HERE ###
# Calcular la mediana de los días desde el onboarding
median_onboarding = df['n_days_after_onboarding'].median()

print(f"The median value of n_days_after_onboarding is: {median_onboarding}")


# Half of the people in the dataset had 40% or more of their sessions in just the last month, yet the overall median time since onboarding is almost five years.
# 
# Make a histogram of `n_days_after_onboarding` for just the people who had 40% or more of their total sessions in the last month.

# In[90]:


# Histogram
### YOUR CODE HERE ###
# 1. Filtrar el dataframe
# Seleccionamos solo a los que tienen 40% o más de sus sesiones en el último mes
high_recent_activity = df[df['percent_sessions_in_last_month'] >= 0.4]

# 2. Crear el histograma
plt.figure(figsize=(12, 6))
sns.histplot(data=high_recent_activity, x='n_days_after_onboarding')

plt.title('Days since onboarding for users with >40% sessions in last month', fontsize=14)
plt.xlabel('Days after onboarding', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()


# The number of days since onboarding for users with 40% or more of their total sessions occurring in just the last month is a uniform distribution. This is very strange. It's worth asking Waze why so many long-time users suddenly used the app so much in the last month.

# ### **Task 3b. Handling outliers**
# 
# The box plots from the previous section indicated that many of these variables have outliers. These outliers do not seem to be data entry errors; they are present because of the right-skewed distributions.
# 
# Depending on what you'll be doing with this data, it may be useful to impute outlying data with more reasonable values. One way of performing this imputation is to set a threshold based on a percentile of the distribution.
# 
# To practice this technique, write a function that calculates the 95th percentile of a given column, then imputes values > the 95th percentile with the value at the 95th percentile.  such as the 95th percentile of the distribution.
# 
# 

# In[92]:


### YOUR CODE HERE ###
def imputar_outliers(columna):
    # 1. Calcular el percentil 95
    percentil_95 = df[columna].quantile(0.95)
    
    # 2. Reemplazar los valores mayores al percentil
    df.loc[df[columna] > percentil_95, columna] = percentil_95
    
    print(f"Columna {columna}: Valores mayores a {percentil_95:.2f} han sido imputados.")

# Ejemplo de uso con una columna:
# imputar_outliers('driven_km_drives')


# Next, apply that function to the following columns:
# * `sessions`
# * `drives`
# * `total_sessions`
# * `driven_km_drives`
# * `duration_minutes_drives`

# In[93]:


### YOUR CODE HERE ###
# 1. Definimos la función de imputación
def impute_outliers(column_name):
    # Calcular el percentil 95
    limit = df[column_name].quantile(0.95)
    
    # Reemplazar valores mayores al límite con el valor del límite
    df.loc[df[column_name] > limit, column_name] = limit
    
    print(f'{column_name:<25} | Límite (P95): {limit:>10.2f}')

# 2. Lista de columnas a procesar
cols_to_impute = [
    'sessions', 
    'drives', 
    'total_sessions', 
    'driven_km_drives', 
    'duration_minutes_drives'
]

# 3. Aplicamos la función en un bucle
print(f"{'Columna':<25} | {'Valor de Corte'}")
print("-" * 45)
for col in cols_to_impute:
    impute_outliers(col)


# Call `describe()` to see if your change worked.

# In[94]:


### YOUR CODE HERE ###
# Llamar a describe() solo para las columnas modificadas
df[['sessions', 'drives', 'total_sessions', 'driven_km_drives', 'duration_minutes_drives']].describe()


# #### **Conclusion**
# 
# Analysis revealed that the overall churn rate is \~17%, and that this rate is consistent between iPhone users and Android users.
# 
# Perhaps you feel that the more deeply you explore the data, the more questions arise. This is not uncommon! In this case, it's worth asking the Waze data team why so many users used the app so much in just the last month.
# 
# Also, EDA has revealed that users who drive very long distances on their driving days are _more_ likely to churn, but users who drive more often are _less_ likely to churn. The reason for this discrepancy is an opportunity for further investigation, and it would be something else to ask the Waze data team about.

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Execute stage.

# ### **Task 4a. Results and evaluation**
# 
# Having built visualizations in Python, what have you learned about the dataset? What other questions have your visualizations uncovered that you should pursue?
# 
# **Pro tip:** Put yourself in your client's perspective. What would they want to know?
# 
# Use the following code fields to pursue any additional EDA based on the visualizations you've already plotted. Also use the space to make sure your visualizations are clean, easily understandable, and accessible.
# 
# **Ask yourself:** Did you consider color, contrast, emphasis, and labeling?
# 
# 

# ==> ENTER YOUR RESPONSE HERE
# 
# I have learned ....
# EN: Churn is not evenly distributed. Users who drive long daily distances (intensity) are more likely to leave, whereas users who drive frequently (15+ days a month) are the most loyal. Also, our dataset consists of very long-term users (median tenure of ~5 years), yet their activity is heavily concentrated in the last month.
# 
# ES: La fuga no está distribuida uniformemente. Los usuarios que conducen largas distancias diarias (intensidad) tienen más probabilidades de irse, mientras que los que conducen con frecuencia (más de 15 días al mes) son los más leales. Además, el dataset está compuesto por usuarios de muy largo plazo (mediana de ~5 años), aunque su actividad está muy concentrada en el último mes.
# 
# My other questions are ....
# EN: Why is the distribution of n_days_after_onboarding uniform for users with high recent activity? Does the total_sessions variable represent the entire life of the account or just a specific recent window? Why does high mileage correlate with churn—is it driver fatigue or app dissatisfaction during long trips?
# 
# ES: ¿Por qué la distribución de días desde el onboarding es uniforme para usuarios con alta actividad reciente? ¿La variable total_sessions representa toda la vida de la cuenta o solo una ventana reciente? ¿Por qué el alto kilometraje se correlaciona con la fuga: es fatiga del conductor o insatisfacción con la app en viajes largos?
# 
# My client would likely want to know ...
# 
# N: What is the "sweet spot" for driving distance that keeps users engaged without causing them to churn? Can we create a campaign to encourage infrequent users to use the app just 2 or 3 more days a month to move them into the "loyalty zone"?
# 
# ES: ¿Cuál es el "punto ideal" de distancia de conducción que mantiene a los usuarios comprometidos sin causar que se vayan? ¿Podemos crear una campaña para incentivar a los usuarios infrecuentes a usar la app solo 2 o 3 días más al mes para moverlos a la "zona de lealtad"?
# 

# Use the following two code blocks (add more blocks if you like) to do additional EDA you feel is important based on the given scenario.

# In[95]:


### YOUR CODE HERE ###
# 1. Crear una matriz de correlación para las variables clave
plt.figure(figsize=(10, 8))
correlation_matrix = df[['sessions', 'drives', 'total_sessions', 'driven_km_drives', 
                         'duration_minutes_drives', 'n_days_after_onboarding']].corr()

# 2. Graficar el Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Driving Variables')
plt.show()


# In[96]:


### YOUR CODE HERE ###
# 1. Calcular kilómetros por hora promedio por usuario
# (Distancia total / Duración total en horas)
df['km_per_hour'] = df['driven_km_drives'] / (df['duration_minutes_drives'] / 60)

# 2. Comparar la velocidad promedio entre los que se quedan y los que se van
plt.figure(figsize=(12, 5))
sns.boxplot(data=df, x='label', y='km_per_hour', showfliers=False)
plt.title('Estimated Average Speed: Retained vs Churned Users')
plt.ylabel('KM per Hour')
plt.show()

print("Velocidad promedio por grupo:")
print(df.groupby('label')['km_per_hour'].mean())


# ### **Task 4b. Conclusion**
# 
# Now that you've explored and visualized your data, the next step is to share your findings with Harriet Hadzic, Waze's Director of Data Analysis. Consider the following questions as you prepare to write your executive summary. Think about key points you may want to share with the team, and what information is most relevant to the user churn project.
# 
# **Questions:**
# 
# 1. What types of distributions did you notice in the variables? What did this tell you about the data?
# 
#    EN: Most driving-related variables (distance, duration, number of drives) are right-skewed. This tells us that while the majority of users perform short, routine trips, there is a small "power user" segment that drives significantly more than the average.
# 
# ES: La mayoría de las variables de conducción (distancia, duración, viajes) tienen un sesgo a la derecha. Esto indica que mientras la mayoría realiza viajes cortos y rutinarios, existe un pequeño segmento de "usuarios intensos" que conduce significativamente más que el promedio.
# 
# 3. Was there anything that led you to believe the data was erroneous or problematic in any way?
# 
#    EN: The uniform distribution of n_days_after_onboarding across all levels of recent activity is highly unusual. It suggests that total_sessions might not be a true lifetime count, but rather data from a specific, limited time window.
# 
# ES: La distribución uniforme de los días desde el onboarding en todos los niveles de actividad reciente es muy inusual. Sugiere que total_sessions podría no ser un conteo histórico real, sino datos de una ventana de tiempo específica y limitada.
# 
# 5. Did your investigation give rise to further questions that you would like to explore or ask the Waze team about?
# 
#    EN: I would ask the Waze team to clarify the exact definition of a "session" versus a "drive" and confirm the data retention policy for the total_sessions variable to ensure our churn model isn't based on truncated history.
# 
# ES: Le pediría al equipo de Waze aclarar la definición exacta de "sesión" frente a "viaje" (drive) y confirmar la política de retención de datos para la variable total_sessions para asegurar que nuestro modelo de fuga no se base en un historial truncado.
# 
# 7. What percentage of users churned and what percentage were retained?
# 
#    EN: Approximately 18% of the users in this dataset churned, while 82% were retained.
# 
# ES: Aproximadamente el 18% de los usuarios en este dataset se fugaron (churn), mientras que el 82% fueron retenidos.
# 
# 9. What factors correlated with user churn? How?
# 
#     EN: Churn is positively correlated with daily distance intensity. Users who drive more kilometers per driving day are more likely to churn. Conversely, higher usage frequency (more driving days per month) is a strong indicator of retention.
# 
# ES: La fuga está positivamente correlacionada con la intensidad de distancia diaria. Los usuarios que recorren más kilómetros por día de conducción tienen más probabilidades de irse. Por el contrario, una mayor frecuencia de uso es un fuerte indicador de retención.
# 
# 11. Did newer uses have greater representation in this dataset than users with longer tenure? How do you know?
# EN: The dataset is dominated by long-term users, with a median tenure of nearly 5 years (~1,741 days). We know this because the distribution of n_days_after_onboarding is uniform and doesn't show a spike in recent sign-ups.
# 
# ES: El dataset está dominado por usuarios de largo plazo, con una mediana de antigüedad de casi 5 años (~1,741 días). Sabemos esto porque la distribución de antigüedad es uniforme y no muestra un pico de registros recientes.

# ==> ENTER YOUR RESPONSES TO QUESTIONS 1-6 HERE
# 
# 1. What types of distributions did you notice in the variables? What did this tell you about the data?
# 
#    EN: Most driving-related variables (distance, duration, number of drives) are right-skewed. This tells us that while the majority of users perform short, routine trips, there is a small "power user" segment that drives significantly more than the average.
# 
# ES: La mayoría de las variables de conducción (distancia, duración, viajes) tienen un sesgo a la derecha. Esto indica que mientras la mayoría realiza viajes cortos y rutinarios, existe un pequeño segmento de "usuarios intensos" que conduce significativamente más que el promedio.
# 
# 3. Was there anything that led you to believe the data was erroneous or problematic in any way?
# 
#    EN: The uniform distribution of n_days_after_onboarding across all levels of recent activity is highly unusual. It suggests that total_sessions might not be a true lifetime count, but rather data from a specific, limited time window.
# 
# ES: La distribución uniforme de los días desde el onboarding en todos los niveles de actividad reciente es muy inusual. Sugiere que total_sessions podría no ser un conteo histórico real, sino datos de una ventana de tiempo específica y limitada.
# 
# 5. Did your investigation give rise to further questions that you would like to explore or ask the Waze team about?
# 
#    EN: I would ask the Waze team to clarify the exact definition of a "session" versus a "drive" and confirm the data retention policy for the total_sessions variable to ensure our churn model isn't based on truncated history.
# 
# ES: Le pediría al equipo de Waze aclarar la definición exacta de "sesión" frente a "viaje" (drive) y confirmar la política de retención de datos para la variable total_sessions para asegurar que nuestro modelo de fuga no se base en un historial truncado.
# 
# 7. What percentage of users churned and what percentage were retained?
# 
#    EN: Approximately 18% of the users in this dataset churned, while 82% were retained.
# 
# ES: Aproximadamente el 18% de los usuarios en este dataset se fugaron (churn), mientras que el 82% fueron retenidos.
# 
# 9. What factors correlated with user churn? How?
# 
#     EN: Churn is positively correlated with daily distance intensity. Users who drive more kilometers per driving day are more likely to churn. Conversely, higher usage frequency (more driving days per month) is a strong indicator of retention.
# 
# ES: La fuga está positivamente correlacionada con la intensidad de distancia diaria. Los usuarios que recorren más kilómetros por día de conducción tienen más probabilidades de irse. Por el contrario, una mayor frecuencia de uso es un fuerte indicador de retención.
# 
# 11. Did newer uses have greater representation in this dataset than users with longer tenure? How do you know?
# EN: The dataset is dominated by long-term users, with a median tenure of nearly 5 years (~1,741 days). We know this because the distribution of n_days_after_onboarding is uniform and doesn't show a spike in recent sign-ups.
# 
# ES: El dataset está dominado por usuarios de largo plazo, con una mediana de antigüedad de casi 5 años (~1,741 días). Sabemos esto porque la distribución de antigüedad es uniforme y no muestra un pico de registros recientes.
# 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
