#!/usr/bin/env python
# coding: utf-8

# # **Waze Project**
# **Course 2 - Get Started with Python**

# Welcome to the Waze Project!
# 
# Your Waze data analytics team is still in the early stages of their user churn project. Previously, you were asked to complete a project proposal by your supervisor, May Santner. You have received notice that your project proposal has been approved and that your team has been given access to Waze's user data. To get clear insights, the user data must be inspected and prepared for the upcoming process of exploratory data analysis (EDA).
# 
# A Python notebook has been prepared to guide you through this project. Answer the questions and create an executive summary for the Waze data team.

# # **Course 2 End-of-course project: Inspect and analyze data**
# 
# In this activity, you will examine data provided and prepare it for analysis. This activity will help ensure the information is,
# 
# 1.   Ready to answer questions and yield insights
# 
# 2.   Ready for visualizations
# 
# 3.   Ready for future hypothesis testing and statistical methods
# <br/>
# 
# **The purpose** of this project is to investigate and understand the data provided.
# 
# **The goal** is to use a dataframe contructed within Python, perform a cursory inspection of the provided dataset, and inform team members of your findings.
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Understand the situation
# * How can you best prepare to understand and organize the provided information?
# 
# **Part 2:** Understand the data
# 
# * Create a pandas dataframe for data learning, future exploratory data analysis (EDA), and statistical activities
# 
# * Compile summary information about the data to inform next steps
# 
# **Part 3:** Understand the variables
# 
# * Use insights from your examination of the summary data to guide deeper investigation into variables
# 
# 
# <br/>
# 
# Follow the instructions and answer the following questions to complete the activity. Then, you will complete an Executive Summary using the questions listed on the PACE Strategy Document.
# 
# Be sure to complete this activity before moving on. The next course item will provide you with a completed exemplar to compare to your own work.
# 
# 

# # **Identify data types and compile summary information**
# 

# <img src="images/Pace.png" width="100" height="100" align=left>
# 
# # **PACE stages**

# Throughout these project notebooks, you'll see references to the problem-solving framework, PACE. The following notebook components are labeled with the respective PACE stages: Plan, Analyze, Construct, and Execute.

# <img src="images/Plan.png" width="100" height="100" align=left>
# 
# 
# ## **PACE: Plan**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response:

# ### **Task 1. Understand the situation**
# 
# *   How can you best prepare to understand and organize the provided driver data?
# 
# 
# *Begin by exploring your dataset and consider reviewing the Data Dictionary.*

# ==> ENTER YOUR RESPONSE HERE
# To best prepare for this project, I will start by conducting a thorough review of the Data Dictionary to ensure I understand the meaning and unit of measurement for each variable (e.g., sessions vs. drives). I will then use Python (Pandas) to perform a preliminary data inspection using functions like .info() to identify data types and missing values, and .describe() to detect outliers in driving distances or durations.
# 
# My goal is to align these technical findings with the business objective: predicting user churn. I will organize the data by grouping key metrics (activity, distance, and device type) to see how they relate to the label column, ensuring the dataset is clean and logically consistent before moving into the Exploratory Data Analysis (EDA) phase.

# <img src="images/Analyze.png" width="100" height="100" align=left>
# 
# ## **PACE: Analyze**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Analyze stage.

# ### **Task 2a. Imports and data loading**
# 
# Start by importing the packages that you will need to load and explore the dataset. Make sure to use the following import statements:
# 
# *   `import pandas as pd`
# 
# *   `import numpy as np`
# 

# In[1]:


# Import packages for data manipulation
### YOUR CODE HERE ###
import pandas as pd

import numpy as np


# Then, load the dataset into a dataframe. Creating a dataframe will help you conduct data manipulation, exploratory data analysis (EDA), and statistical activities.
# 
# **Note:** As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# Load dataset into dataframe
df = pd.read_csv('waze_dataset.csv')


# ### **Task 2b. Summary information**
# 
# View and inspect summary information about the dataframe by **coding the following:**
# 
# 1.   df.head(10)
# 2.   df.info()
# 
# *Consider the following questions:*
# 
# 1. When reviewing the `df.head()` output, are there any variables that have missing values?
# In the first 10 rows, no missing values are visible. However, I need to check df.info() to see if there are nulls in the entire dataset of 14,999 rows.
# 2. When reviewing the `df.info()` output, what are the data types? How many rows and columns do you have?
# The dataset consists of 3 float64 columns, 8 int64 columns, and 2 object columns (label and device).
# here are 14,999 rows (entries) and 13 columns.
# 
# 3. Does the dataset have any missing values?
# Yes. The label column has only 14,299 non-null values, meaning there are 700 missing values in this variable.

# In[3]:


### YOUR CODE HERE ###
# Display the first 10 rows to get a visual sense of the data and check for immediate NaN values.
# Visualizar las primeras 10 filas para tener una idea de los datos y buscar valores nulos inmediatos.

df.head(10)


# In[4]:


### YOUR CODE HERE ###
# Check data types (Dtypes) and identify columns with missing values (non-null counts).
# Verificar los tipos de datos (Dtypes) e identificar columnas con valores faltantes (conteos no nulos).

df.info()


# ==> ENTER YOUR RESPONSES TO QUESTIONS 1-3 HERE
# 1. When reviewing the `df.head()` output, are there any variables that have missing values?
# In the first 10 rows, no missing values are visible. However, I need to check df.info() to see if there are nulls in the entire dataset of 14,999 rows.
# 2. When reviewing the `df.info()` output, what are the data types? How many rows and columns do you have?
# The dataset consists of 3 float64 columns, 8 int64 columns, and 2 object columns (label and device).
# here are 14,999 rows (entries) and 13 columns.
# 
# 3. Does the dataset have any missing values?
# Yes. The label column has only 14,299 non-null values, meaning there are 700 missing values in this variable.
# 
# The label column is the only one with missing data (700 nulls). We will need to decide whether to drop these rows or impute them before building the machine learning model."

# ### **Task 2c. Null values and summary statistics**
# 
# Compare the summary statistics of the 700 rows that are missing labels with summary statistics of the rows that are not missing any values.
# 
# **Question:** Is there a discernible difference between the two populations?
# 

# In[5]:


# Isolate rows with null values
### YOUR CODE HERE ###
# Aislar las filas con valores nulos

null_df = df[df['label'].isnull()]

# Display summary stats of rows with null values
### YOUR CODE HERE ###
# Mostrar estadísticas descriptivas de las filas con valores nulos

null_df.describe()


# In[6]:


# Isolate rows without null values
# Aislar las filas SIN valores nulos (para comparar)
### YOUR CODE HERE ###

not_null_df = df[df['label'].notnull()]

# Display summary stats of rows without null values
### YOUR CODE HERE ###

# Mostrar estadísticas descriptivas de las filas sin valores nulos
not_null_df.describe()


# ==> ENTER YOUR RESPONSE HERE
# Question: Is there a discernible difference between the two populations?
# 
# Comparing the summary statistics between the two populations reveals no discernible difference. The average number of sessions, driving days, and kilometers driven are nearly identical for both users with missing labels and those without. This indicates that the missingness in the 'label' column is likely random and does not introduce a specific bias toward a certain type of user.

# ### **Task 2d. Null values - device counts**
# 
# Next, check the two populations with respect to the `device` variable.
# 
# **Question:** How many iPhone users had null values and how many Android users had null values?

# In[7]:


# Get count of null values by device
### YOUR CODE HERE ###

# Obtener el conteo de valores nulos por dispositivo
null_df['device'].value_counts()


# ==> ENTER YOUR RESPONSE HERE
# 
# Question: How many iPhone users had null values and how many Android users had null values?
# 
# There are 447 iPhone users and 253 Android users with null values. This represents a roughly 64% to 36% split, which matches the overall device distribution in the dataset. This further confirms that the missing data is not related to the type of device being used.

# Now, of the rows with null values, calculate the percentage with each device&mdash;Android and iPhone. You can do this directly with the [`value_counts()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html) function.

# In[8]:


# Calculate % of iPhone nulls and Android nulls
### YOUR CODE HERE ###
# Calcular el % de nulos en iPhone y Android

null_df['device'].value_counts(normalize=True)


# How does this compare to the device ratio in the full dataset?
# 
# The proportions are nearly identical. The full dataset has a ratio of approximately 64.5% iPhone and 35.5% Android, while the null-labeled rows show 63.9% iPhone and 36.1% Android. This confirms that the missing data is not biased by device type and is likely missing at random (MCAR).

# In[9]:


# Calculate % of iPhone users and Android users in full dataset
### YOUR CODE HERE ###
# Calcular el % de usuarios de iPhone y Android en el dataset completo

df['device'].value_counts(normalize=True)


# The percentage of missing values by each device is consistent with their representation in the data overall.
# 
# There is nothing to suggest a non-random cause of the missing data.

# Examine the counts and percentages of users who churned vs. those who were retained. How many of each group are represented in the data?
# 
# There are 11,763 retained users and 2,536 churned users represented in the dataset. Proportionally, 82.26% of users were retained, while 17.74% churned. This indicates a baseline churn rate that we will aim to predict and understand through further analysis.

# In[11]:


# Calculate counts of churned vs. retained
### YOUR CODE HERE ###
# Obtener los conteos de usuarios que abandonaron vs. retenidos

print(not_null_df['label'].value_counts())

# Obtener los porcentajes de usuarios que abandonaron vs. retenidos
print("\nPercentages:")
print(not_null_df['label'].value_counts(normalize=True) * 100)


# This dataset contains 82% retained users and 18% churned users.
# 
# Next, compare the medians of each variable for churned and retained users. The reason for calculating the median and not the mean is that you don't want outliers to unduly affect the portrayal of a typical user. Notice, for example, that the maximum value in the `driven_km_drives` column is 21,183 km. That's more than half the circumference of the earth!

# In[12]:


# Calculate median values of all columns for churned and retained users
### YOUR CODE HERE ###
# Calcular los valores de la mediana de todas las columnas para los usuarios que abandonaron y los retenidos

not_null_df.groupby('label').median(numeric_only=True)


# This offers an interesting snapshot of the two groups, churned vs. retained:
# 
# Users who churned averaged ~3 more drives in the last month than retained users, but retained users used the app on over twice as many days as churned users in the same time period.
# 
# The median churned user drove ~200 more kilometers and 2.5 more hours during the last month than the median retained user.
# 
# It seems that churned users had more drives in fewer days, and their trips were farther and longer in duration. Perhaps this is suggestive of a user profile. Continue exploring!

# Calculate the median kilometers per drive in the last month for both retained and churned users.
# 
# Begin by dividing the `driven_km_drives` column by the `drives` column. Then, group the results by churned/retained and calculate the median km/drive of each group.

# In[13]:


# Add a column to df called `km_per_drive`
### YOUR CODE HERE ###
# Añadir una columna llamada 'km_per_drive' dividiendo km totales por viajes

df['km_per_drive'] = df['driven_km_drives'] / df['drives']

# Group by `label`, calculate the median, and isolate for km per drive
### YOUR CODE HERE ###
# Agrupar por etiqueta, calcular la mediana y aislar la columna km_per_drive

df.groupby('label')[['km_per_drive']].median()


# The median retained user drove about one more kilometer per drive than the median churned user. How many kilometers per driving day was this?
# 
# To calculate this statistic, repeat the steps above using `driving_days` instead of `drives`.

# In[14]:


# Add a column to df called `km_per_driving_day`
### YOUR CODE HERE ###

# Añadir una columna llamada 'km_per_driving_day'
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']

# Group by `label`, calculate the median, and isolate for km per driving day
### YOUR CODE HERE ###
# Agrupar por etiqueta, calcular la mediana y aislar la columna km_per_driving_day

df.groupby('label')[['km_per_driving_day']].median()



# Now, calculate the median number of drives per driving day for each group.

# In[15]:


# Add a column to df called `drives_per_driving_day`
### YOUR CODE HERE ###

# Añadir una columna llamada 'drives_per_driving_day'
df['drives_per_driving_day'] = df['drives'] / df['driving_days']

# Group by `label`, calculate the median, and isolate for drives per driving day
### YOUR CODE HERE ###

# Agrupar por 'label', calcular la mediana y aislar la nueva columna
df.groupby('label')[['drives_per_driving_day']].median()


# The median user who churned drove 698 kilometers each day they drove last month, which is almost ~240% the per-drive-day distance of retained users. The median churned user had a similarly disproporionate number of drives per drive day compared to retained users.
# 
# It is clear from these figures that, regardless of whether a user churned or not, the users represented in this data are serious drivers! It would probably be safe to assume that this data does not represent typical drivers at large. Perhaps the data&mdash;and in particular the sample of churned users&mdash;contains a high proportion of long-haul truckers.
# 
# In consideration of how much these users drive, it would be worthwhile to recommend to Waze that they gather more data on these super-drivers. It's possible that the reason for their driving so much is also the reason why the Waze app does not meet their specific set of needs, which may differ from the needs of a more typical driver, such as a commuter.

# Finally, examine whether there is an imbalance in how many users churned by device type.
# 
# Begin by getting the overall counts of each device type for each group, churned and retained.

# In[16]:


# For each label, calculate the number of Android users and iPhone users
### YOUR CODE HERE ###

# Para cada etiqueta, calcular el número de usuarios de Android y de iPhone
df.groupby('label')['device'].value_counts()


# Now, within each group, churned and retained, calculate what percent was Android and what percent was iPhone.

# In[17]:


# For each label, calculate the percentage of Android users and iPhone users
### YOUR CODE HERE ###
# Para cada etiqueta, calcular el porcentaje de usuarios de Android y de iPhone
df.groupby('label')['device'].value_counts(normalize=True)


# The ratio of iPhone users and Android users is consistent between the churned group and the retained group, and those ratios are both consistent with the ratio found in the overall dataset.

# <img src="images/Construct.png" width="100" height="100" align=left>
# 
# ## **PACE: Construct**
# 
# **Note**: The Construct stage does not apply to this workflow. The PACE framework can be adapted to fit the specific requirements of any project.
# 
# 

# <img src="images/Execute.png" width="100" height="100" align=left>
# 
# ## **PACE: Execute**
# 
# Consider the questions in your PACE Strategy Document and those below to craft your response:

# ### **Task 3. Conclusion**
# 
# Recall that your supervisor, May Santer, asked you to share your findings with the data team in an executive summary. Consider the following questions as you prepare to write your summary. Think about key points you may want to share with the team, and what information is most relevant to the user churn project.
# 
# **Questions:**
# 
# 1. Did the data contain any missing values? How many, and which variables were affected? Was there a pattern to the missing data?
# 
# 2. What is a benefit of using the median value of a sample instead of the mean?
# 
# 3. Did your investigation give rise to further questions that you would like to explore or ask the Waze team about?
# 
# 4. What percentage of the users in the dataset were Android users and what percentage were iPhone users?
# 
# 5. What were some distinguishing characteristics of users who churned vs. users who were retained?
# 
# 6. Was there an appreciable difference in churn rate between iPhone users vs. Android users?
# 
# 
# 
# 

# ==> ENTER YOUR RESPONSES TO QUESTIONS 1-6 HERE
# 
# 1. Did the data contain any missing values? How many, and which variables were affected? Was there a pattern to the missing data?
# 
# 
# Yes, the data contained 700 missing values, exclusively in the label (churn vs. retained) variable. After investigating, no clear pattern was found; the missing values were distributed proportionately across device types (63.9% iPhone / 36.1% Android), which is consistent with the overall dataset. This suggests the data is Missing Completely at Random (MCAR).
# 
# 2. What is a benefit of using the median value of a sample instead of the mean?
# 
# The median is a robust statistic, meaning it is not influenced by extreme outliers. In this dataset, we saw a user with over 21,000 km driven in a month. A mean would be pulled upward by such extreme values, giving a distorted view of a "typical" user, whereas the median provides a more accurate representation of the center of the distribution.
# 
# 3. Did your investigation give rise to further questions?
# 
# Absolutely. The most pressing question is: Why are "super-drivers" churning? We found that users who leave drive nearly 700 km per driving day. It would be worth asking the Waze team if the app has specific pain points for long-haul drivers (e.g., lack of specialized routing for large vehicles, or battery/data consumption issues during 10+ hour shifts).
# 
# 4. What percentage of the users in the dataset were Android users and what percentage were iPhone users?
# 
# The distribution was approximately 64.5% iPhone users and 35.5% Android users.
# 
# 5. What were some distinguishing characteristics of users who churned vs. users who were retained?
# 
# 
# Intensity: Churned users drove significantly more kilometers per driving day (~697 km vs. ~289 km) and had more drives per driving day (10 vs. 4).
# 
# Frequency: Interestingly, churned users were active on fewer days (median of 6 days) compared to retained users (median of 14 days).
# 
# Profile: Churned users appear to be high-intensity, short-duration users (potential professional drivers), while retained users show a more consistent, "commuter-style" usage pattern.
# 
# 6. Was there an appreciable difference in churn rate between iPhone users vs. Android users?
# 
# No. The churn rate was nearly identical across both platforms (~17.7% for iPhone and ~17.5% for Android). This indicates that the reason for churning is likely behavioral or related to the service itself, rather than a technical issue specific to one mobile operating system.
# 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
