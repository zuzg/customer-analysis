import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans


def plot_transactions_in_time(df: pd.DataFrame) -> None:
    """
    Plot transactions count and average by days
    """
    transactions_agg = df.groupby(
        df.TransactionDate)["TransactionValue"].agg(
        TransactionMean='mean', TransactionCount='count')

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("Transactions in time", fontsize=16)
    ax[0].bar(transactions_agg.index, transactions_agg.TransactionCount, color="green")
    ax[0].set_title("Count")
    ax[1].bar(transactions_agg.index, transactions_agg.TransactionMean)
    ax[1].set_title("Average")
    ax[1].set_xlabel("month")
    fig.show()


def plot_customers_scatter(df: pd.DataFrame) -> None:
    """
    Show every client separately - transaction sums and counts
    """
    customers_agg = df.groupby([df.CustomerId, df.CustomerType, df.CustomerChannel])[
        "TransactionValue"].agg(TransactionSum='sum', TransactionCount='count')
    customers_agg['Customer info'] = customers_agg.index
    customers_agg[['CustomerId', 'CustomerType', 'CustomerChannel']] = pd.DataFrame(
        customers_agg['Customer info'].tolist(), index=customers_agg.index)
    
    fig = px.scatter(customers_agg, x=customers_agg.TransactionCount, y=customers_agg.TransactionSum,
                     symbol="CustomerType", color="CustomerChannel", opacity=0.5, hover_data=['CustomerId'])
    fig.update_layout(title="Customers's transactions")
    fig.show()

    print("Greatest count of transactions")
    print(f"id: {customers_agg.TransactionCount.argmax()} count: {max(customers_agg.TransactionCount)}")
    print("Greatest sum of transactions")
    print(f"id: {customers_agg.TransactionSum.argmax()} sum: {max(customers_agg.TransactionSum)}")


def plot_alcohol_classes(df: pd.DataFrame) -> None:
    """
    Show proportions of classes of alcohol bought on pie chart
    """
    alco_classes = df.groupby(df.AlcoholClass)["TransactionValue"].agg(TransactionMean='mean', TransactionCount='count')
    fig, ax = plt.subplots()
    ax.pie(alco_classes.TransactionCount, labels=alco_classes.index, autopct='%1.1f%%')
    ax.set_title("Percentage of classes of alcohol bought")
    plt.show()


def plot_customer_status(df: pd.DataFrame) -> None:
    """
    Plot statistics with respect to customer status
    """
    customer_status = df.groupby([df.CustomerStatus])["TransactionValue"].agg(
        TransactionSum='sum', TransactionMean='mean', TransactionCount='count')

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].bar(customer_status.index, customer_status.TransactionCount, color="green")
    ax[0].set_title("Count")
    ax[1].bar(customer_status.index, customer_status.TransactionMean)
    ax[1].set_title("Average")
    ax[2].bar(customer_status.index, customer_status.TransactionSum, color="orange")
    ax[2].set_title("Sum")
    plt.show()


def plot_customer_map(df: pd.DataFrame) -> None:
    """
    Show customer transactions on map
    """
    fig = px.density_mapbox(df, lat='CustomerLatitude', lon='CustomerLongitude', z='TransactionValue', radius=8,
                            center=dict(lat=52, lon=18), zoom=5, mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def plot_elbow(df: pd.DataFrame) -> None:
    """
    Visualize elbow method for clusters
    """
    errors = []
    for k in range(1, 11):
        model = KMeans(n_clusters=k, random_state=23)
        model.fit(df)
        errors.append(model.inertia_)

    plt.title('The Elbow Method')
    plt.xlabel('k')
    plt.ylabel('Error of Cluster')
    sns.pointplot(x=list(range(1, 11)), y=errors)
    plt.show()
