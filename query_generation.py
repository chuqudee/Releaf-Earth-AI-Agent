import re

def generate_sql_query(user_input):
    """
    Converts user input into an SQL query based on predefined patterns.
    """
    user_input = user_input.lower()
    
    if "total p1 sourced" in user_input:
        return "SELECT SUM(`p1 supplied in tons`) AS total_p1_sourced FROM P1_Purchase;"
    
    elif "average p1 price per ton" in user_input:
        return "SELECT SUM(`p1 supplied in tons` * `p1 price per ton`) / SUM(`p1 supplied in tons`) AS avg_p1_price FROM P1_Purchase;"
    
    elif "unique suppliers" in user_input:
        return "SELECT COUNT(DISTINCT name) AS unique_suppliers FROM P1_Purchase;"
    
    elif "number of transactions" in user_input:
        return "SELECT COUNT(*) AS num_transactions FROM P1_Purchase;"
    
    elif "average volume per supplier" in user_input:
        return "SELECT SUM(`p1 supplied in tons`) / COUNT(DISTINCT name) AS avg_volume FROM P1_Purchase;"
    
    elif "average amount per supplier" in user_input:
        return "SELECT SUM(`p1 supplied in tons` * `p1 price per ton`) / COUNT(DISTINCT name) AS avg_amount FROM P1_Purchase;"
    
    elif "supply per agent" in user_input:
        return "SELECT rp, SUM(`p1 supplied in tons`) AS total_supplied FROM P1_Purchase GROUP BY rp;"
    
    elif "average p1 price per ton by agent" in user_input:
        return "SELECT rp, SUM(`p1 supplied in tons` * `p1 price per ton`) / SUM(`p1 supplied in tons`) AS avg_price FROM P1_Purchase GROUP BY rp;"
    
    elif "p1 moved" in user_input:
        return "SELECT SUM(quantity) AS p1_moved FROM P1_Delivery;"
    
    elif "number of trips" in user_input:
        return "SELECT COUNT(*) AS num_trips FROM P1_Delivery;"
    
    elif "trip per lmv" in user_input:
        return "SELECT lmv, COUNT(*) AS trips FROM P1_Delivery GROUP BY lmv;"
    
    else:
        return "Unknown query request. Please refine your input."

# Example usage
if __name__ == "__main__":
    user_query = input("Enter your question: ")
    sql_query = generate_sql_query(user_query)
    print("Generated SQL Query:", sql_query)
