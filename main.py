import logging
from app import app
from query_handler import DatabaseHandler
from analysis import DataAnalysis
from report_generator import ReportGenerator
from maintenance_prediction import PredictiveMaintenance
from agents import CrewAIAgent
from llm_integration import OpenAILLM

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    """
    Initializes all components and starts the Flask server.
    """
    logging.info("Starting Releaf AI System...")

    # Initialize core components
    db_handler = DatabaseHandler()
    data_analysis = DataAnalysis()
    report_generator = ReportGenerator()
    maintenance_predictor = PredictiveMaintenance()
    ai_agents = CrewAIAgent()
    llm = OpenAILLM(model="gpt-4o")  # Default model can be changed

    logging.info("All components initialized successfully.")
    
    # Start the Flask server
    logging.info("Launching Flask API...")
    app.run(debug=True, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()
