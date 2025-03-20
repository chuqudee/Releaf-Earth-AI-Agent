import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_data(query):
    """Fetches data from the database and returns a Pandas DataFrame."""
    conn = sqlite3.connect("releaf_earth.db")
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def plot_p1_sourcing_trends():
    """Plots trends in P1 sourcing over time."""
    query = "SELECT date, SUM(amount_ton) AS total_p1_sourced FROM P1_Purchase GROUP BY date ORDER BY date;"
    df = fetch_data(query)
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df['date'], y=df['total_p1_sourced'])
    plt.xlabel("Date")
    plt.ylabel("Total P1 Sourced (Tons)")
    plt.title("P1 Sourcing Trends Over Time")
    plt.xticks(rotation=45)
    plt.show()

def plot_lmv_utilization():
    """Plots LMV utilization metrics."""
    query = """
        SELECT LMV, SUM(p1_moved) AS total_p1_moved, SUM(distance_km) AS total_distance 
        FROM P1_Delivery GROUP BY LMV;
    """
    df = fetch_data(query)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=df['LMV'], y=df['total_p1_moved'], color='b', ax=ax1)
    ax1.set_ylabel("Total P1 Moved (Tons)", color='b')
    ax2 = ax1.twinx()
    sns.lineplot(x=df['LMV'], y=df['total_distance'], color='r', marker="o", ax=ax2)
    ax2.set_ylabel("Total Distance (KM)", color='r')
    plt.title("LMV Utilization: P1 Moved vs Distance Traveled")
    plt.show()
