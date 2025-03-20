import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def format_response(data, response_type="table"):
    """
    Formats the response into different formats: table, chart, report, or metric values.
    
    :param data: Query result (list of tuples or Pandas DataFrame)
    :param response_type: "table", "chart", "report", or "metric"
    :return: Formatted response
    """
    if response_type == "table":
        return format_as_table(data)
    elif response_type == "chart":
        return format_as_chart(data)
    elif response_type == "report":
        return format_as_report(data)
    elif response_type == "metric":
        return format_as_metric(data)
    else:
        raise ValueError("Invalid response type. Choose from 'table', 'chart', 'report', or 'metric'.")

def format_as_table(data):
    """Formats data as a structured table."""
    if isinstance(data, pd.DataFrame):
        return data.to_string(index=False)
    else:
        return pd.DataFrame(data).to_string(index=False)

def format_as_chart(data):
    """Generates a visualization from the data using Matplotlib/Seaborn."""
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)
    
    if len(df.columns) >= 2:
        plt.figure(figsize=(8, 5))
        sns.barplot(x=df.iloc[:, 0], y=df.iloc[:, 1])
        plt.xticks(rotation=45)
        plt.title("Data Visualization")
        plt.show()
        return "Chart displayed successfully."
    else:
        return "Insufficient data to generate a chart."

def format_as_report(data):
    """Generates a textual report based on the data."""
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)
    
    report = f"Report Summary:\nTotal Records: {len(df)}\n"
    report += f"Columns: {', '.join(df.columns)}\n"
    report += f"First 5 rows:\n{df.head().to_string(index=False)}"
    return report

def format_as_metric(data):
    """Extracts and returns numerical summary metrics from the data."""
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)
    
    return df.describe().to_string()

if __name__ == "__main__":
    sample_data = [("A", 10), ("B", 20), ("C", 30)]
    df_sample = pd.DataFrame(sample_data, columns=["Category", "Value"])
    
    print(format_response(df_sample, "table"))
    print(format_response(df_sample, "report"))
    format_response(df_sample, "chart")
