import pandas as pd
import numpy as np
import re


def prepare_data(X: pd.DataFrame):
    """
    Preprocess training data.
    Parameter
    ----------
    X: pd.DataFrame
        The loaded data
    Returns
    -------
    A clean, preprocessed version of the data
    """
    df = X.copy()  # Work on a copy to avoid modifying the original DataFrame

    # Define columns to remove
    COL_TO_REMOVE = ["Cre_date", "Repub_date", "Area", "City", "Pic_num", "Supply_score", "Test"]

    # Remove irrelevant columns if they exist
    df = df.drop([col for col in COL_TO_REMOVE if col in df.columns], axis=1, errors='ignore')

    # Convert columns to appropriate types
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    if 'Km' in df.columns:
        df['Km'] = pd.to_numeric(df['Km'].replace(r'[^\d.]', '', regex=True), errors='coerce')

    if 'capacity_Engine' in df.columns:
        df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'].replace(r'[^\d.]', '', regex=True), errors='coerce')

    # Handle missing values in 'Curr_ownership' and 'Prev_ownership'
    if 'Curr_ownership' in df.columns:
        df['Curr_ownership'] = df['Curr_ownership'].replace('', 'לא מוגדר').fillna('לא מוגדר')
    if 'Prev_ownership' in df.columns:
        df['Prev_ownership'] = df['Prev_ownership'].replace('', 'לא מוגדר').fillna('לא מוגדר')

    # Ownership history
    if 'Prev_ownership' in df.columns and 'Curr_ownership' in df.columns:
        df['Ownership_Change'] = (df['Prev_ownership'] != df['Curr_ownership']).astype(int)
        df['Is_Private'] = ((df['Curr_ownership'] == 'פרטית') & (df['Prev_ownership'] == 'פרטית')).astype(int)


    # Convert 'capacity_Engine' to numeric
    if 'capacity_Engine' in df.columns:
        df['capacity_Engine'] = df['capacity_Engine'].astype(str).str.replace(',', '')
        df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')
        # Fill missing 'capacity_Engine' with the median
        df['capacity_Engine'] = df.groupby('manufactor')['capacity_Engine'].transform(lambda x: x.fillna(x.median()))
        overall_median = df['capacity_Engine'].median()
        df['capacity_Engine'].fillna(overall_median, inplace=True)
        # Ensure 'capacity_Engine' is of type int
        df['capacity_Engine'] = df['capacity_Engine'].astype(int)

    # Filter by condition if 'model' and 'capacity_Engine' columns are present
    if 'model' in df.columns and 'capacity_Engine' in df.columns:
        model_capacity_map = {
            # Your existing map...
        }
        df = filter_by_condtion(df, model_capacity_map)

    # Fill missing 'Engine_type' with the mode
    if 'Engine_type' in df.columns:
        if not df['Engine_type'].mode().empty:
            df['Engine_type'].fillna(df['Engine_type'].mode()[0], inplace=True)

    # Fill missing 'Color' with mode within groups
    if 'Color' in df.columns:
        if 'manufactor' in df.columns and 'model' in df.columns:
            df['Color'] = df.groupby(['manufactor', 'model'])['Color'].transform(
                lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Unknown'))
        df = color_mapping(df)

    # Apply corrections to 'model' column if 'model' column is present
    if 'model' in df.columns:
        df['model'] = df['model'].apply(remove_year_from_model)
        df['model'] = df['model'].apply(apply_corrections)

    # Creating a new 'Horsepower' column by applying the estimate_horsepower function if required columns are present
    if 'capacity_Engine' in df.columns and 'Engine_type' in df.columns:
        df['Horsepower'] = df.apply(lambda row: estimate_horsepower(row['capacity_Engine'], row['Engine_type']), axis=1)

    # Transmission type
    if 'Gear' in df.columns:
        df['Is_Automatic'] = (df['Gear'] == 'אוטומטית').astype(int)
    # df = filter_columns(df)

    return df


def check_keywords(description):
    # returns a dictionary where each keyword is a key, and the value is 1 if the keyword is found in the description
    # (case-insensitive), or 0 if not.
    keywords =['שמור', 'תאונה', 'מטופל', 'חדש']

    return {keyword: int(keyword in str(description).lower()) for keyword in keywords}


# Create Horsepower column
def estimate_horsepower(capacity, engine_type):
    if pd.isna(capacity):
        return np.nan
    if engine_type == 'בנזין':  # Petrol
        return capacity * 0.1
    elif engine_type == 'דיזל':  # Diesel
        return capacity * 0.08
    elif engine_type == 'חשמלי':  # Electric
        return capacity * 1.34  # Assuming capacity is in kW for electric
    else:
        return capacity * 0.09  # Default multiplier for unknown engine types


# This function provides a rough estimate of horsepower based on engine capacity and type. It creates a standardized
# 'Horsepower' feature, which can be more informative than separate 'capacity' and 'engine_type' features.


# Function that corrects the model names
def apply_corrections(model):
    corrections = [
        ('אודי', 'אאודי'),
        ('סקודה אוקטביה ספייס\s*', 'סקודה אוקטביה ספייס'),
        ('אאודי A(\d)', r'אאודי A\1'),
        ('סקודה פאביה\s*', 'סקודה פאביה'),
        ('סקודה רומסטר\s*', 'סקודה רומסטר'),
        ('אאודי RS(\d)', r'אאודי RS\1'),
        ('ג`טה', 'ג\'טה'),
        ('ג\'אז', 'ג`אז'),
        ('אוונסיס', 'אונסיס'),
        ('גראנד, וויאג`ר', 'גראנד, וויאג\'ר'),
    ]
    for pattern, replacement in corrections:
        model = re.sub(pattern, replacement, model)
    return model


# Removes the year from model names
def remove_year_from_model(model):
    return re.sub(r'\b20\d{2}\b', '', model)


def filter_by_condtion(df, model_capacity_map):
    def matches_condition(model, capacity):
        if model in model_capacity_map:
            condition = model_capacity_map[model]
            if isinstance(condition, dict):
                return capacity in condition.values()
            else:
                return capacity == condition
        return True

    return df[df.apply(lambda row: matches_condition(row['model'], row['capacity_Engine']), axis=1)]


def filter_columns(df):
    columns_to_keep = [
        'Year', 'Km', 'capacity_Engine', 'Curr_ownership', 'Prev_ownership',
        'Description', 'manufactor', 'model', 'Hand', 'Gear', 'Engine_type',
        'Color'
    ]
    return df[columns_to_keep]


def color_mapping(df):
    # Define the mapping of colors to simplified categories
    color_map = {
        'לבן': 'White',
        'שחור': 'Dark',
        'כסף': 'Silver',
        'אפור': 'Grey',
        'כחול': 'Blue',
        'אדום': 'Red',
        'ירוק': 'Green',
        'צהוב': 'Yellow',
        'כתום': 'Orange',
        'חום': 'Brown',
        'בורדו': 'Red',
        'סגול': 'Purple',
        'זהב': 'Gold',
        'פלטינום': 'Silver',
        'אחר': 'Other',
        'לילך': 'Purple',
        'טורקיז': 'Blue',
        'פנינה': 'white',
        'בז`': 'Beige',
        'תכלת': 'Blue',
        'קרם': 'White',
        'ורוד': 'Pink',
        'אינדיגו': 'Blue',
        'יין': 'Red'
    }
    df['Color'] = df['Color'].map(color_map).fillna('Other')
    return df
