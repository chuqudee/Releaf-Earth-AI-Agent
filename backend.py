import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from crewai import Crew, Agent, Task
from openai import OpenAI

# Database Setup
def create_database():
    conn = sqlite3.connect("releaf_earth.db")
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''CREATE TABLE IF NOT EXISTS P1_Purchase (
                        supplier_name TEXT,
                        location TEXT,
                        amount_ton REAL,
                        amount_paid REAL,
                        date TEXT,
                        lmv_used TEXT,
                        distance_km REAL)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS P1_Delivery (
                        lmv_id TEXT,
                        delivery_date TEXT,
                        amount_ton REAL)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS Fueling_Records (
                        lmv_id TEXT,
                        fuel_issued REAL,
                        date TEXT)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS LMV_Maintenance (
                        lmv_id TEXT,
                        maintenance_type TEXT,
                        cost REAL,
                        date TEXT,
                        distance_traveled_km REAL,
                        fuel_used_liters REAL,
                        load_weight_tons REAL)''')
    
    conn.commit()
    conn.close()

def load_csv_to_sql():
    conn = sqlite3.connect("releaf_earth.db")
    
    files = {
        "P1_Purchase": "P1_Purchase.csv",
        "P1_Delivery": "P1_Delivery.csv",
        "Fueling_Records": "Fueling_Records.csv",
        "LMV_Maintenance": "LMV_Maintenance_Tracker.csv"
    }
    
    for table, file in files.items():
        df = pd.read_csv(file)
        df.to_sql(table, conn, if_exists="replace", index=False)
    
    conn.close()

# Define AI Agents
class SeniorDatabaseDeveloper(Agent):
    def process_query(self, user_query):
        sql_query = self.convert_to_sql(user_query)
        return query_database(sql_query)
    
    def convert_to_sql(self, user_query):
        return "SELECT * FROM P1_Purchase LIMIT 5;"

class SeniorDataAnalyst(Agent):
    def analyze_data(self, df):
        insights = {
            "total_p1_sourced": df["amount_ton"].sum(),
            "average_price_per_ton": df["amount_paid"].sum() / df["amount_ton"].sum()
        }
        return insights
    
    def generate_chart(self, df):
        plt.figure(figsize=(10,5))
        sns.barplot(x=df['supplier_name'], y=df['amount_ton'])
        plt.xticks(rotation=45)
        plt.title("P1 Sourced per Supplier")
        plt.xlabel("Supplier")
        plt.ylabel("Tons")
        plt.savefig("p1_sourced_chart.png")
        plt.close()
        return "p1_sourced_chart.png"
    
    def predict_maintenance(self, df):
        features = df[["distance_traveled_km", "fuel_used_liters", "load_weight_tons"]]
        target = df["maintenance_required"]
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return predictions

class SeniorReportEditor(Agent):
    def generate_report(self, insights, chart_path, maintenance_predictions):
        report = f"Total P1 Sourced: {insights['total_p1_sourced']} Tons\n"
        report += f"Average Price/Ton: {insights['average_price_per_ton']} Naira\n"
        report += f"Chart saved at: {chart_path}\n"
        report += f"Predicted Maintenance Needs: {maintenance_predictions[:5]} (sample)\n"
        return report

# Query Function
def query_database(sql_query):
    conn = sqlite3.connect("releaf_earth.db")
    df = pd.read_sql_query(sql_query, conn)
    conn.close()
    return df

# Run Agent Pipeline
def run_agent_pipeline(user_query):
    db_agent = SeniorDatabaseDeveloper()
    analyst_agent = SeniorDataAnalyst()
    report_agent = SeniorReportEditor()
    
    df = db_agent.process_query(user_query)
    insights = analyst_agent.analyze_data(df)
    chart_path = analyst_agent.generate_chart(df)
    maintenance_predictions = analyst_agent.predict_maintenance(df)
    report = report_agent.generate_report(insights, chart_path, maintenance_predictions)
    
    return report

if __name__ == "__main__":
    create_database()
    load_csv_to_sql()
    user_query = "Show me P1 purchase data"
    result = run_agent_pipeline(user_query)
    print(result)
