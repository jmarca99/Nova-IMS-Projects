
# Importing the termcolor library
import myimports as imps
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import seaborn as sns #0.12.2
import matplotlib.pyplot as plt #3.7.1
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway
from sklearn.tree import DecisionTreeClassifier

###########################################################################################################################################

# Defining the summary_stats function
def summary_stats(df):
    
    print(imps.colored('HEAD OF DATASET:', attrs=['bold']))
    display(df.head())
    
    print(imps.colored('\nTAIL OF DATASET:', attrs=['bold']))
    display(df.tail())
    
    print(imps.colored('\nINFORMATION PANEL:', attrs=['bold']))
    display(df.info())
    
    print(imps.colored('\nDATASET DESCRIPTION:', attrs=['bold']))
    display(df.describe().T)
    
    print(imps.colored('\nROW DUPLICATED:', attrs=['bold']), df.duplicated().sum())
    
    print(imps.colored('\nMISSING VALUES:', attrs=['bold']))
    print(df.isnull().sum())

    print(imps.colored('\nMISSING VALUES in %:', attrs=['bold']))
    print(round(df.isnull().sum() / len(df) * 100 ,2))

###########################################################################################################################################

# Defining the print_unique_counts function
def print_unique_counts(df, col):
    unique_counts = df[col].value_counts()
    total_count = unique_counts.sum()
    print(f"{col} unique values:\n")
    
    for value, count in unique_counts.items():
        percentage = (count / total_count) * 100
        print(f"{value}: {count} ({percentage:.2f}%)")

###########################################################################################################################################

# Defining the confusion matrix function
def confus_matrix(y_valid, predictions):
    # Get unique labels from predictions
    labels = np.unique(predictions)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_valid, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
   ###########################################################################################################################################
# SUB-FUNCTION
# Function to get the first X rows for each process
def get_first_x_rows_per_process(df, process_col, x):
    return df.groupby(process_col).head(x)

###########################################################################################################################################
# SUB-FUNCTION
# Define a function to flatten lists
def flatten_lists(series):
    flattened_list = []
    for value in series:
        # Verificar se o valor não é NaN e é uma lista antes de tentar iterar sobre ele
        if isinstance(value, list):
            flattened_list.extend(value)
    return list(set(flattened_list))

###########################################################################################################################################
# SUB-FUNCTION
# Define function to search if the series is = to Y
def has_outsourcer(series):
    return int(any(value == "Y" for value in series))

###########################################################################################################################################
# SUB-FUNCTION
# Define a função para calcular a diferença entre o ano presente e o valor na coluna "OrgUnitSince"
def calculate_mean_difference(series):
    # Calcular a diferença entre o ano presente (2024) e cada valor na série
    differences = 2024 - series
    # Calcular a média das diferenças para o grupo
    mean_difference = differences.mean()
    return mean_difference

###########################################################################################################################################

# Define funtion to test if list lenght is bigger than 0
def is_list_length_higher_than_zero(lst):
    return (len(lst) > 0)

###########################################################################################################################################

# Define function to create buckets
def prep_bucket(dataset, dataset2, outcome_dataset, z):
    # Actual data
    df_moment = get_first_x_rows_per_process(dataset,"Request Identifier", z)
    df_moment = df_moment.drop_duplicates(subset="Request Identifier", keep='last')
    df_moment = df_moment[["Request Identifier","Task arrival date"]]
    
    # Apply the function
    df = get_first_x_rows_per_process(dataset,"Request Identifier", z-1)

    # Group by process
    df = df.groupby("Request Identifier").agg({
        "Task arrival date": "min",
        "Task execution end date": "max",
        "Actvity ID": lambda x: list(x),
        "idBPMApplicationAction": lambda x: list(x),
        "Time surpassed": "sum",
        "Task Executer": lambda x: list(x), 
        "Task executer department": "nunique", 
        "Age": "mean",
        "OrgUnitSince": lambda x: calculate_mean_difference(x), # mean time of all executers inside that role
        "idBPMRequirement": lambda x: flatten_lists(x), 
        "IsOutSourcer": lambda x: has_outsourcer(x) #means that part of the process was out sourced
    })

    # Reset index
    df = df.reset_index()

    # Rename columns
    df = df.rename(columns={
        "Task arrival date": "Task arrival date",
        "Task execution end date": "Task execution end date",
        "Actvity ID": "Activity ID List",
        "idBPMApplicationAction": "Actions List",
        "Time surpassed": "Total Time surpassed (hours)",
        "Task Executer": "Executer List",
        "Task executer department": "Number of Depart. Involved",
        "Age": "Mean Age of Executers",
        "OrgUnitSince": "Mean Time on Department",
        "IsOutSourcer": "Out-Sourced Involved",
        "idBPMRequirement": "All BPM Requirements"
    })

    # Fill missing (due to task executers) as 0 (in mean age and mean time on department)
    df["Mean Age of Executers"].fillna(0, inplace=True)
    df["Mean Time on Department"].fillna(0, inplace=True)
    
    # FEATURE ENGINEERING: Arrival Date of Moment
    df = df.merge(df_moment, how="left", on="Request Identifier")

    # FEATURE ENGINEERING: Duration in hours
    df["Duration (hours)"] = df["Task execution end date"] - df["Task arrival date_x"] # first arrival
    df["Duration (hours)"] = df["Duration (hours)"].apply(lambda x: x.total_seconds()/60/60)

    # FEATURE ENGINEERING: Hour of Arrival in Moment Z
    df['Hour Task Arrival Moment'] = df['Task arrival date_y'].dt.hour

    # FEATURE ENGINEERING: Month of Arrival in Moment Z
    df['Month Task Arrival Moment'] = df['Task arrival date_y'].dt.month
    
    # FEATYRE ENGINEERING: Number of Executers
    df["Number of Executers"] = df['Executer List'].apply(lambda x: len(set(x)))

    # FEATURE ENGINEERING: Does it have BPM Requirements?
    df['W/ BPM Requirements'] = (df['All BPM Requirements'].apply(is_list_length_higher_than_zero)).astype("int")

    # FEATURE ENGINEERING: Activity Index
    for i in range(df['Activity ID List'].apply(len).max()):
        col_name = f'Activity_{i+1}'
        # Extract the activity at the current position if it exists, otherwise use None
        df[col_name] = df['Activity ID List'].apply(lambda x: x[i] if len(x) > i else None)

    # FEATURE ENGINEERING: Action Index
    for i in range(df['Actions List'].apply(len).max()):
        col_name = f'Action_{i+1}'
        df[col_name] = df['Actions List'].apply(lambda x: x[i] if len(x) > i else None)
        
    # FEATURE ENGINEERING: Action Index
    for i in range(df['Executer List'].apply(len).max()):
        col_name = f'Executer_{i+1}'
        df[col_name] = df['Executer List'].apply(lambda x: x[i] if len(x) > i else None)
        
    # FEATURE ADDING: Add Specific Request
    df = df.merge(dataset2, how="left", on="Request Identifier")
    
    # Convert the 'Date' column to datetime type
    df["ID_203"] = pd.to_datetime(df["ID_203"], utc=True)
    # Then convert to timezone-naive by removing the timezone information
    df["ID_203"] = df["ID_203"].dt.tz_localize(None)

    # FEATURE ENGINEERING: Calculate Time to Start Process
    df["Time to Start Process"] =  df['Task arrival date_x'] - df["ID_203"]
    df["Time to Start Process"] = df["Time to Start Process"].apply(lambda x: x.total_seconds()/60/60)    
    
    # Add OUTCOME
    df = df.merge(outcome_dataset[["Request Identifier", "Outcome"]], on="Request Identifier", how="left")

    # Remove Unecessary Columns
    df.drop(["Task execution end date","Task arrival date_x","Task arrival date_y", "All BPM Requirements", "ID_203", 
             "Request Identifier", #drop index
             "Activity ID List","Actions List", "Executer List"], inplace=True, axis=1) # drop lists
    df.drop(["Activity_1", "Action_1"], inplace=True, axis=1) # Remove first index since they are the same for all
    
    return df

###########################################################################################################################################

# Define function to transform values into categories
def convert_to_categorical(df, categorical_columns):
    # Convert specified columns to categorical if they exist in the DataFrame
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df

###########################################################################################################################################

# Define function to get spearman correlation
def get_spearman_corr(df):
    # Calculate the Spearman correlation matrix
    corr = df.corr(method='spearman')

    # Mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Create correlation matrix heatmap
    fig , ax = plt.subplots(figsize=(20, 16))  # Increase the figure size
    heatmap = sns.heatmap(corr, 
                          mask=mask,
                          square=True,
                          linewidths=.5,
                          cmap='coolwarm',
                          cbar_kws={'shrink': .4,'ticks': [-1, -.5, 0, 0.5, 1]},
                          fmt='.2f',
                          vmin=-1,
                          vmax=1,
                          annot=True, 
                          annot_kws={'size': 8})  # Adjust annotation size

    # Decoration
    plt.title(f"Spearman correlation matrix", fontsize=15, y=1.03)
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})
    plt.show()

###########################################################################################################################################

# Define function to get cramer's scores
def get_cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

###########################################################################################################################################

# Define function to get ANOVA scores
def get_anova(df, categorical_col, numerical_col):
    groups = df.groupby(categorical_col)[numerical_col].apply(list)
    return f_oneway(*groups)

###########################################################################################################################################

# Define function to get model importance
def get_model_importance_dt(df, target):
    # Assuming 'prefix_5' is your DataFrame and 'Outcome' is your target variable
    X = df.drop(columns=[target])
    y = df[target]

    # Fit the Decision Tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)

    # Get feature importances
    feature_importances = model.feature_importances_

    # Create a DataFrame for better visualization
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Sort the DataFrame by importance for better visualization
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances as a horizontal bar plot
    plt.figure(figsize=(12, 10))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.xlabel('Importance')
    plt.title('Feature Importances')

    # Adding labels to each bar
    for index, value in enumerate(importance_df['Importance']):
        plt.text(value + 0.01, index, f'{value:.4f}', va='center')

    plt.show()

###########################################################################################################################################















