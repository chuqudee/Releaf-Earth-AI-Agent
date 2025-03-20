document.addEventListener("DOMContentLoaded", function () {
    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query-input");
    const modelSelect = document.getElementById("model-select");
    const resultsContainer = document.getElementById("results");
    const visualizeBtn = document.getElementById("visualize-btn");
    const maintenanceBtn = document.getElementById("maintenance-btn");
    
    queryForm.addEventListener("submit", async function (event) {
        event.preventDefault();
        const query = queryInput.value;
        const model = modelSelect.value;
        
        resultsContainer.innerHTML = "<p>Loading...</p>";
        
        try {
            const response = await fetch("/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, model })
            });
            const data = await response.json();
            resultsContainer.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        } catch (error) {
            resultsContainer.innerHTML = "<p>Error fetching data.</p>";
        }
    });

    visualizeBtn.addEventListener("click", async function () {
        resultsContainer.innerHTML = "<p>Generating visualization...</p>";
        try {
            const response = await fetch("/visualize", { method: "POST" });
            const data = await response.json();
            resultsContainer.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Visualization">`;
        } catch (error) {
            resultsContainer.innerHTML = "<p>Error generating visualization.</p>";
        }
    });

    maintenanceBtn.addEventListener("click", async function () {
        resultsContainer.innerHTML = "<p>Predicting maintenance needs...</p>";
        try {
            const response = await fetch("/predict-maintenance", { method: "POST" });
            const data = await response.json();
            resultsContainer.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        } catch (error) {
            resultsContainer.innerHTML = "<p>Error predicting maintenance.</p>";
        }
    });
});
