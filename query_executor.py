import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

db_path = "releaf_earth.db"

def execute_query(db_path, query):
    """
    Executes a given SQL query and returns the result as a Pandas DataFrame.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def format_response(df, response_type="table"):
    """
    Formats query results based on the response type.
    - "table": Returns a Pandas DataFrame
    - "chart": Generates a Seaborn chart
    - "summary": Provides a text-based summary
    """
    if response_type == "table":
        return df
    elif response_type == "chart":
        if df.shape[1] < 2:
            return "Insufficient data for chart generation."
        plt.figure(figsize=(8, 5))
        sns.barplot(x=df.columns[0], y=df.columns[1], data=df)
        plt.xticks(rotation=45)
        plt.title("Generated Chart")
        plt.show()
    elif response_type == "summary":
        return df.describe().to_string()
    else:
        return "Invalid response type specified."

# Example usage
db_path = "releaf_earth.db"
query = "SELECT * FROM P1_Purchase LIMIT 5;"
df_result = execute_query(db_path, query)
print(format_response(df_result, response_type="table"))