# app_fraud.py

import gradio as gr
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the Model and Scaler ---
model = joblib.load('fraud_detection_model.joblib')
scaler = joblib.load('fraud_scaler.joblib')

# --- 2. Define the Custom CSS for a Modern UI ---
modern_css = """
/* Overall Body and Background */
body {
    background-color: #1a1a1a;
    color: #ffffff;
    font-family: 'Roboto', sans-serif;
}
/* Main container for the Gradio app */
#fraud-detection-app {
    border: none;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    border-radius: 10px;
    background: #2b2b2b;
}
/* Header Title */
#header .text-center {
    color: #ffffff;
    font-size: 2.5em;
    font-weight: bold;
    background: -webkit-linear-gradient(45deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
/* Input/Output Textboxes */
.gradio-container .form-control {
    background-color: #3c3c3c;
    color: #ffffff;
    border: 1px solid #555;
    border-radius: 8px;
}
/* The 'Detect Fraud' Button */
#detect-button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px 0 rgba(0, 198, 255, 0.5);
}
#detect-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px 0 rgba(0, 198, 255, 0.7);
}
/* Output Textbox Styling */
#output-textbox textarea {
    font-size: 1.2em;
    font-weight: bold;
    text-align: center;
}
/* Custom classes for conditional styling */
.fraud-alert {
    background-color: #8B0000 !important; /* Dark Red */
    color: white !important;
}
.safe-alert {
    background-color: #006400 !important; /* Dark Green */
    color: white !important;
}
"""

# --- 3. Define the Prediction Function ---
# Because there are 30 features, we'll accept a comma-separated string of values.
def predict_fraud(transaction_data_str):
    if not transaction_data_str:
        return gr.update(value="Error: Please provide transaction data.", elem_classes=["fraud-alert"])

    try:
        # Convert comma-separated string to a numpy array of floats
        values = np.array([float(x.strip()) for x in transaction_data_str.split(',')])

        if len(values) != 30:
            return gr.update(
                value=f"Error: Expected 30 feature values, but got {len(values)}.",
                elem_classes=["fraud-alert"]
            )
            
        # Create a DataFrame with the correct feature names
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        input_df = pd.DataFrame([values], columns=feature_names)

        # Scale the Time and Amount columns
        cols_to_scale = ['Time', 'Amount']
        input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1] # Probability of fraud

        # Return result with conditional styling
        if prediction[0] == 1:
            result_text = f"🚨 FRAUD DETECTED 🚨\nProbability: {probability:.2%}"
            return gr.update(value=result_text, elem_classes=["fraud-alert"])
        else:
            result_text = f"✅ Transaction Appears Safe ✅\nProbability of Fraud: {probability:.2%}"
            return gr.update(value=result_text, elem_classes=["safe-alert"])

    except Exception as e:
        return gr.update(value=f"An error occurred: {str(e)}", elem_classes=["fraud-alert"])


# --- 4. Build the Gradio App with gr.Blocks ---
with gr.Blocks(css=modern_css, elem_id="fraud-detection-app") as app:
    with gr.Column(elem_id="header"):
        gr.Markdown("<h1 style='text-align: center;'>Fraud Detection Terminal</h1>")

    gr.Markdown("Enter the 30 transaction features as a comma-separated string to check for fraud.")

    # Input for comma-separated values
    transaction_input = gr.Textbox(
        label="Transaction Features (V1, V2, ..., V28, Time, Amount)",
        placeholder="e.g., -1.35, 1.25, ..., 86400, 59.99"
    )

    # Themed button
    detect_button = gr.Button("Detect Fraud", elem_id="detect-button")

    # Output with conditional styling
    output_textbox = gr.Textbox(label="Detection Result", elem_id="output-textbox", interactive=False)

    # Connect the button to the prediction function
    detect_button.click(
        fn=predict_fraud,
        inputs=transaction_input,
        outputs=output_textbox
    )

app.launch()