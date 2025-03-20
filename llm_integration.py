import openai
import sqlite3
from query_generation import generate_sql_query
from query_executor import execute_query

# Load OpenAI API Key (Hardcoded)
OPENAI_API_KEY = "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"

# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def llm_generate_query(user_input):
    """
    Uses OpenAI's LLM to convert natural language input into an SQL query.
    """
    prompt = f"Convert the following user request into an SQL query: {user_input}"

    response = client.chat.completions.create(
        model="gpt-4",  # Change to gpt-3.5-turbo if needed
        messages=[
            {"role": "system", "content": "You are a helpful AI that generates SQL queries."},
            {"role": "user", "content": prompt}
        ]
    )

    sql_query = response.choices[0].message.content.strip()
    return sql_query

def process_user_request(user_input):
    """
    Processes user input by generating an SQL query, executing it, and formatting the result.
    """
    sql_query = llm_generate_query(user_input)
    print(f"Generated SQL Query: {sql_query}")

    result = execute_query(sql_query)
    return result

if __name__ == "__main__":
    user_input = input("Enter your query: ")
    output = process_user_request(user_input)
    print("Result:", output)
