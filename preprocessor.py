import pandas as pd

def preprocess(df,region_df):
    # filtering for summer olympics
    df = df[df['Season'] == 'Summer']
    # merge with region_df
    df = df.merge(region_df, on='NOC', how='left')
    # dropping duplicates
    df.drop_duplicates(inplace=True)
    # one hot encoding medals
    df = pd.concat([df, pd.get_dummies(df['Medal'])], axis=1)
    return df
def preprocess_medal_prediction(df):
    # Create a binary target variable where 1 indicates a medal was won
    df['Medal_Won'] = df['Medal'].apply(lambda x: 1 if pd.notna(x) else 0)  # 1 if medal, 0 if none
    
    # Select relevant columns (Athlete's age, sport, country/region, and whether they won a medal)
    df = df[['Age', 'Sport', 'region', 'Medal_Won']]
    
    # Remove rows with missing data
    df = df.dropna()
    
    return df