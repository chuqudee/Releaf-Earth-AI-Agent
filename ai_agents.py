from crewai import Agent
from query_handler import execute_query
from analysis import analyze_data
from visualization import generate_chart
from maintenance_prediction import train_model, predict_maintenance
from openai import OpenAI
import os

# Load OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4o")

# Train the maintenance prediction model
trained_model = train_model()

# Senior Database Developer - Converts CSV to SQL and executes queries
senior_db_developer = Agent(
    name="Senior Database Developer",
    role="Handles SQL database queries and conversions",
    description="This agent is responsible for retrieving data from the structured database based on user prompts.",
    function=lambda query: execute_query(query)
)

# Senior Data Analyst - Analyzes data, finds insights, and calculates metrics
senior_data_analyst = Agent(
    name="Senior Data Analyst",
    role="Performs data analysis and metric calculations",
    description="This agent interprets data, calculates performance metrics, and provides insights based on user requests.",
    function=lambda query: analyze_data(query)
)

# Senior Report Editor - Summarizes query results and provides recommendations
senior_report_editor = Agent(
    name="Senior Report Editor",
    role="Generates structured reports and recommendations",
    description="This agent summarizes data analysis results into a user-friendly format, including key insights and actionable recommendations.",
    function=lambda analysis: llm.generate_response(f"Summarize and provide insights: {analysis}")
)

# LMV Maintenance Predictor - Predicts LMV faults and recommends preventive actions
lmv_maintenance_predictor = Agent(
    name="LMV Maintenance Predictor",
    role="Predicts potential LMV faults based on maintenance trends",
    description="This agent forecasts potential LMV issues using machine learning and recommends preventive measures.",
    function=lambda distance, fuel, load, prev_maint_days: predict_maintenance(
        trained_model, distance, fuel, load, prev_maint_days
    )
)

# Visualization Generator - Creates charts based on data
visualization_generator = Agent(
    name="Visualization Generator",
    role="Generates visual representations of data",
    description="This agent creates charts and graphs to visualize data trends and patterns.",
    function=lambda query: generate_chart(query)
)

agents = [
    senior_db_developer,
    senior_data_analyst,
    senior_report_editor,
    lmv_maintenance_predictor,
    visualization_generator
]
