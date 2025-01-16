from flask import Flask, render_template, send_file
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import io
import pandas as pd
import numpy as np

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
columns = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data = pd.read_csv(url, header=None, names=columns)

# Preprocess dataset
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
X = data.iloc[:, 2:]
y = data['diagnosis']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

app = Flask(__name__)

@app.route('/')
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Logistic Regression Model Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Train Data Shape: {X_train.shape}", ln=True)
    pdf.cell(200, 10, txt=f"Test Data Shape: {X_test.shape}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Model Accuracy: {accuracy:.2f}", ln=True)
    
    pdf_output = "logistic_regression_report.pdf"
    pdf.output(pdf_output)  # Save the PDF to a file
    
    return send_file(pdf_output, as_attachment=True, download_name='logistic_regression_report.pdf')

if __name__ == '__main__':
    app.run(debug=True)
    
    

