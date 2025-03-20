import sqlite3
import pandas as pd

# Define CSV file paths
csv_files = {
    "P1_Purchase": "data/P1_Purchase.csv",
    "P1_Delivery": "data/P1_Delivery.csv",
    "Machinery_Expenses": "data/Machinery_Expenses.csv",
    "LMV_Maintenance_Tracker": "data/LMV_Maintenance_Tracker.csv"
}

# Create SQLite database
conn = sqlite3.connect("releaf_earth.db")
cursor = conn.cursor()

# Define table schemas
table_schemas = {
    "P1_Purchase": """
    CREATE TABLE IF NOT EXISTS P1_Purchase (
        date TEXT, rp TEXT, name TEXT, gender TEXT, supplier_type TEXT, lga TEXT,
        phone_number TEXT, community TEXT, p1_supplied_in_tons REAL, p1_price_per_ton REAL,
        grade TEXT, lmv_used TEXT, week_code TEXT, distance_covered REAL
    )""",
    "P1_Delivery": """
    CREATE TABLE IF NOT EXISTS P1_Delivery (
        date TEXT, ticket_no TEXT, source TEXT, items TEXT, uom TEXT, description TEXT,
        quantity REAL, lmv TEXT, lmv_capacity REAL, pick_up_location TEXT, coordinates TEXT,
        discovery_id TEXT, week_code TEXT
    )""",
    "Machinery_Expenses": """
    CREATE TABLE IF NOT EXISTS Machinery_Expenses (
        date TEXT, ticket_no TEXT, receiver_department TEXT, item TEXT, measurement TEXT,
        description TEXT, quantity REAL, unit_cost REAL, total_cost REAL, weeks TEXT
    )""",
    "LMV_Maintenance_Tracker": """
    CREATE TABLE IF NOT EXISTS LMV_Maintenance_Tracker (
        issue_id TEXT, date_reported TEXT, reporter TEXT, lmv TEXT, issue_description TEXT,
        type_of_issue TEXT, type_of_maintenance TEXT, diagnosis TEXT, maintenance_actions TEXT,
        replace_parts TEXT, parts_to_be_replaced TEXT, cost_of_parts REAL, cost_of_maintenance REAL,
        total_cost REAL, technician TEXT, start_date_of_maintenance TEXT, end_date_of_maintenance TEXT,
        time_on_down_time REAL, maintenance_status TEXT, lmv_status TEXT, remarks TEXT, week TEXT
    )"""
}

# Create tables
for table, schema in table_schemas.items():
    cursor.execute(schema)

# Load CSV data into tables
for table, file_path in csv_files.items():
    df = pd.read_csv(file_path)
    df.to_sql(table, conn, if_exists='replace', index=False)

# Commit and close connection
conn.commit()
conn.close()

print("CSV files successfully converted to SQLite database.")
