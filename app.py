from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import plotly.express as px
import json
import openai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import os
from dotenv import load_dotenv

# Access environment variables
#app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')  # Fallback if not set
openai.api_key = os.environ.get('OPENAI_API_KEY')

app = Flask(__name__)

# Database setup
db_path = "releaf_earth.db"

# Function to generate SQL query using OpenAI
def generate_sql_query(user_input):
    # Define the database schema in the prompt
    schema_info = """
    You have access to a database containing the following tables:
    - P1_Purchase: date, rp, name, gender, supplier_type, lga, phone_number, community, p1_supplied_in_tons, p1_price_per_ton, grade, lmv_used, week_code, distance_covered
    - P1_Delivery: date, ticket_no, source, items, uom, description, quantity, lmv, lmv_capacity, pick_up_location, coordinates, discovery_id, week_code
    - Machinery_Expenses: date, ticket_no, receiver_department, item, measurement, description, quantity, unit_cost, total_cost, week_code
    - LMV_Maintenance_Tracker: issue_id, date_reported, reporter, lmv, issue_description, type_of_issue, type_of_maintenance, diagnosis, maintenance_actions, replace_parts, parts_to_be_replaced, cost_of_parts, cost_of_maintenance, total_cost, technician, start_date_of_maintenance, end_date_of_maintenance, time_on_down_time, maintenance_status, lmv_status, remarks, week
    Please go through the database and understand what each and every table and column represents.

    P1 means palm nut, and it is measured in tons using the "p1_supplied_in_tons" column. this column can be used to calculate tonnage per distance covered (KM) in the P1_Purchase table.
    The distance covered for every delivery trip is recorded in P1_Purchase table, in the "distance_covered" column. Use JOIN query to retrieve if neccessary.
    no need to use JOIN to calculate average kilometers per trip.

    Please note that the "date" column is an actual date datatype column, but the "week_code" column is a text datatype column. every value in the "week_code" column is passes as a string value.
    Be sure to understand what the "week_code" column values stand for. for example; 
    "Jan wk 1 - 24" means the first week of January in 2024.
    "Apr wk 4 - 23" simply means the 4th week of April in 2024
    "Feb wk 3 - 25" also means the 3rd week of February in 2025

    In the P1_Delivery table, the "quantity" column contains the amount of P1(in tons) that was delivered. it can be used to calculate tonage per trip. for example;
    SUM(quantity) / COUNT(ticket_no)

    You should only use the "week_code" column whenever the word "week" is mentioned in an input, otherwise, use the "date" column.
    Fuel expenses are recorded in the machinery_expenses table. they can be seen as "Fuel" in the "item" column. they are measured in litres in the "measurement" column. there is also "quantity" and "cost" column.
    Diesel expenses are recorded in the machinery_expenses table. they can be seen as "Diesel" in the "item" column. they are measured in litres in the "measurement" column. 

    Accurately Convert the following user request into a valid SQLite-compatible SQL query:
    {user_input}

    When creating a JOIN sql query from user input, be sure to specify the table and column name in SELECT, to avoid ambiguity. for example;
    SELECT P1_Purchase.week_code, SUM(Machinery_Expenses.total_cost) / SUM(P1_Purchase.p1_supplied_in_tons) AS cost_per_ton
    FROM Machinery_Expenses
    JOIN P1_Purchase ON Machinery_Expenses.date = P1_Purchase.date
    WHERE item = 'Fuel'
    GROUP BY P1_Purchase.week_code
    ORDER BY cost_per_ton DESC
    LIMIT 1;

    Respond ONLY with the SQL query.
    """.format(user_input=user_input)
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "assistant", "content": schema_info}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating SQL query: {str(e)}"

# Function to execute SQL query
def execute_query(sql_query):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql_query, conn)

# Function to generate insights
def generate_insights(df):
    insights = []
    
    if df.shape[1] == 2 and df.columns[0].lower() == 'department':
        max_expense = df.iloc[:, 1].max()
        max_department = df.loc[df.iloc[:, 1].idxmax(), df.columns[0]]
        insights.append(f"{max_department} had the highest expense of {max_expense}. This could indicate a significant investment or operational cost in this department.")
        
        if max_expense > 10000:
            insights.append(f"The high expense in {max_department} might be due to increased operational costs or investments. Further review is recommended.")
    
    return '\n'.join(insights)

# Function to format response
def format_response(df):
    response = {'text': '', 'chart_data': None, 'insight': ''}
    
    if len(df) == 1 and df.shape[1] == 1:
        response['text'] = f"Result: {df.iloc[0,0]}"
    else:
        response['text'] = df.to_html(index=False, classes='table table-striped')
        
    response['insight'] = generate_insights(df)
    
    if df.shape[1] >= 2:
        try:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1])
            response['chart_data'] = json.loads(fig.to_json())
        except Exception as e:
            response['text'] += f"\nChart error: {str(e)}"
    
    return response

def generate_insight_agent(sql_query, result_df):
    """Agent that generates natural language insights from SQL and results"""
    prompt = f"""
    Analyze this SQL query and its results to provide professional business insights:
    
    SQL Query: {sql_query}
    
    Query Results:
    {result_df.to_string()}
    
    Provide a concise summary with:
    1. Key numerical findings
    2. Notable patterns/trends
    3. Data quality observations
    4. Business implications
    5. Recommended actions
    Use clear, non-technical language.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Insight generation error: {str(e)}"
    

# Function to predict maintenance
def predict_maintenance(distance_covered, time_on_down_time):
    try:
        with open('maintenance_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        # Load data from LMV_Maintenance_Tracker
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT distance_covered, time_on_down_time, maintenance_status FROM LMV_Maintenance_Tracker", conn)
        conn.close()
        
        # Prepare data
        df['maintenance_status'] = df['maintenance_status'].apply(lambda x: 1 if x == 'Completed' else 0)
        X = df[['distance_covered', 'time_on_down_time']]
        y = df['maintenance_status']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Save model
        with open('maintenance_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    # Predict
    prediction = model.predict(np.array([[distance_covered, time_on_down_time]]))
    
    return "Maintenance needed" if prediction[0] == 1 else "No maintenance needed"

# Function to predict P1 price per ton
def predict_p1_price(date, name, supplier_type, lga):
    try:
        with open('p1_price_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        # Load data from P1_Purchase
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT date, name, supplier_type, lga, p1_price_per_ton FROM P1_Purchase", conn)
        conn.close()
        
        # Prepare data
        X = df[['date', 'name', 'supplier_type', 'lga']]
        y = df['p1_price_per_ton']
        
        # Convert categorical variables
        X['name'] = pd.Categorical(X['name']).codes
        X['supplier_type'] = pd.Categorical(X['supplier_type']).codes
        X['lga'] = pd.Categorical(X['lga']).codes
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save model
        with open('p1_price_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    # Predict
    # Note: You need to convert input data to match the model's input format
    # For simplicity, this example assumes direct input conversion is possible
    prediction = model.predict(np.array([[date, name, supplier_type, lga]]))
    
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

# Modify the existing ask route
@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_input = request.json.get('query')
        
        if not user_input:
            return jsonify({"error": "No query provided"}), 400
        
        # Generate and execute SQL
        sql_query = generate_sql_query(user_input)
        results = execute_query(sql_query)
        
        # Format initial response
        formatted = format_response(results)
        
        # Generate AI-powered insight
        insight = generate_insight_agent(sql_query, results)
        
        return jsonify({
            "query": sql_query,
            "result": formatted['text'],
            "chart_data": formatted['chart_data'],
            "insight": insight  # Add the generated insight
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        prediction_type = data.get('prediction_type')
        
        if prediction_type == 'maintenance':
            distance_covered = float(data.get('distance_covered'))
            time_on_down_time = float(data.get('time_on_down_time'))
            return jsonify({"prediction": predict_maintenance(distance_covered, time_on_down_time)})
        elif prediction_type == 'p1_price':
            date = data.get('date')
            name = data.get('name')
            supplier_type = data.get('supplier_type')
            lga = data.get('lga')
            return jsonify({"prediction": predict_p1_price(date, name, supplier_type, lga)})
        else:
            return jsonify({"error": "Invalid prediction type"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
