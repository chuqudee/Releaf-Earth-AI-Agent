import json
from analysis import DataAnalysis

class ReportGenerator:
    def __init__(self):
        """Initialize with a DataAnalysis instance."""
        self.analysis = DataAnalysis()

    def generate_report(self):
        """Generate a structured report containing metrics, tables, and recommendations."""
        report = {}

        # Fetch metrics
        metrics = self.analysis.calculate_metrics()
        report["Key Metrics"] = metrics

        # Fetch summary table
        summary_table = self.analysis.generate_summary_table()
        report["Summary Table"] = summary_table.to_dict(orient="records")

        # Fetch chart (fuel consumption per LMV)
        chart_img = self.analysis.generate_chart()
        report["Fuel Consumption Chart"] = f"data:image/png;base64,{chart_img}"

        # Generate recommendations
        recommendations = self.generate_recommendations(metrics)
        report["Recommendations"] = recommendations

        return report

    def generate_recommendations(self, metrics):
        """Generate data-driven recommendations based on the analysis results."""
        recommendations = []

        # Cost-saving recommendations
        if metrics["Total Purchase Cost"] > 500000:
            recommendations.append("Optimize sourcing strategies to reduce costs.")

        # Fuel efficiency recommendations
        if any(fuel["total_fuel"] > 1000 for fuel in metrics["Fuel Consumption per LMV"]):
            recommendations.append("Consider preventive maintenance for LMVs with high fuel usage.")

        # Delivery improvements
        if metrics["Average Delivery Size (tons)"] < 5:
            recommendations.append("Increase delivery volume per trip to optimize logistics.")

        return recommendations

if __name__ == "__main__":
    report_gen = ReportGenerator()
    report = report_gen.generate_report()

    # Print the report in JSON format
    print(json.dumps(report, indent=4))
