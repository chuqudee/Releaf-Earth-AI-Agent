import pandas as pd

# Dictionary containing CSV file names and paths
csv_files = {
    "P1_Purchase": "data/P1_Purchase.csv",
    "P1_Delivery": "data/P1_Delivery.csv",
    "Machinery_Expenses": "data/Machinery_Expenses.csv",
    "LMV_Maintenance_Tracker": "data/LMV_Maintenance_Tracker.csv"
}

# Process each CSV file
for name, file_path in csv_files.items():
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Rename columns by replacing spaces with underscores
    df.columns = [col.replace(" ", "_") for col in df.columns]
    
    # Save the updated CSV file (overwrite the original file)
    df.to_csv(file_path, index=False)
    
    print(f"Updated column names and saved: {file_path}")