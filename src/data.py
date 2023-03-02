import pandas as pd
from ydata_profiling import ProfileReport


def read_data(filename: str) -> pd.DataFrame:
    """
    Read dataframe from file and adjust columns
    """
    df = pd.read_csv(filename, sep=';', decimal=',')
    df.TransactionDate = pd.to_datetime(df.TransactionDate)  # we can omit exact time as it is always 00:00
    df.head()
    return df


def generate_report(df: pd.DataFrame) -> None:
    """
    Generate ydata profile
    """
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file("data_profile.html")


def split_alcohol_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split alcohol string to name, class and id
    """
    new_df = df.copy()
    new_df[["AlcoholName", "AlcoholClass", "AlcoholId"]] = new_df.ProductName.str.split(expand=True)
    return new_df


def one_hot_encode(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Apply one-hot encoding for chosen column
    """
    one_hot = pd.get_dummies(df[column_name])
    new_data = pd.concat([df.CustomerId, one_hot], axis=1)
    return new_data


def aggregate_by_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate dataframe by CustomerId, apply one-hot encoding for categorical variables
    """
    data = df[['CustomerId', 'TransactionValue']]
    data_agg = data.groupby('CustomerId').agg({'TransactionValue': ['count', 'sum']})
    columns_one_hot = ["AlcoholName", "CustomerType", "CustomerChannel"]

    for col in columns_one_hot:
        one_hot = one_hot_encode(df, col)
        if col == "AlcoholName":
            one_hot_agg = one_hot.groupby('CustomerId').sum()
        else:
            one_hot_agg = one_hot.groupby('CustomerId').first()
        data_agg = data_agg.merge(one_hot_agg, on='CustomerId')

    data_agg.columns = ["TransactionCount", "TransactionSum"] + data_agg.columns[2:].tolist()
    return data_agg


def aggregate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for apriori algorithm
    """
    alcohols = split_alcohol_column(df)
    alcohols['Alcohol'] = alcohols['AlcoholName'] + alcohols['AlcoholClass']
    one_hot = one_hot_encode(alcohols, "Alcohol")
    products = pd.concat([df.TransactionDate, one_hot], axis=1)

    # we are in interested whether a product was bought, not how many times
    # therefore aggregating by max, which will be 1 or 0
    transactions = products.groupby([products.CustomerId, products.TransactionDate]).max()
    return transactions
