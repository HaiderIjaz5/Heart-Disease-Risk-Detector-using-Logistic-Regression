# 🩺 Heart Disease Risk Checker

A Machine Learning project that predicts the likelihood of heart disease using a **Logistic Regression** model. It features an interactive **Gradio UI** for easy user input and real-time prediction — all built inside a Jupyter Notebook.

---

## 🚀 Features

- ✅ Built using `LogisticRegression` from **scikit-learn**
- 🧾 Inputs required:
  - Age (Years)
  - Blood Pressure (mmHg)
  - Cholesterol (mg/dL)
  - Diabetes (Yes/No)
- 📈 Predicts:
  - ✅ Likely Healthy
  - ⚠️ At Risk of Heart Disease
- 🖥 Simple and intuitive Gradio interface with sliders and radio buttons
- 📓 Fully self-contained in a single Jupyter Notebook (`Heart_Disease_Predictor.ipynb`)

---

## 📊 Dataset

- **File**: `heart_disease_dataset.csv`
- **Features**: `age`, `bp`, `cholesterol`, `diabetes`
- **Target**: `target`  
  - `1` = Heart Disease  
  - `0` = Healthy

---

## 🧠 Model Training

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

# Load dataset
df = pd.read_csv("heart_disease_dataset.csv")

# Feature and target selection
X = df[['age', 'bp', 'cholesterol', 'diabetes']]
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

---

## 🖥 Gradio Web Interface

```python
import gradio as gr
import pandas as pd

def predict_heart_disease(age, bp, cholesterol, diabetes):
    diabetes = 1 if diabetes == "Yes" else 0
    input_data = pd.DataFrame([[age, bp, cholesterol, diabetes]], columns=['age', 'bp', 'cholesterol', 'diabetes'])
    prediction = model.predict(input_data)[0]
    return "⚠️ At Risk of Heart Disease" if prediction == 1 else "✅ Likely Healthy"

gr.Interface(
    fn=predict_heart_disease,
    inputs=[
        gr.Slider(10, 100, step=1, label="Age (Years)"),
        gr.Slider(80, 200, step=1, label="Blood Pressure (mmHg)"),
        gr.Slider(100, 350, step=1, label="Cholesterol (mg/dL)"),
        gr.Radio(["No", "Yes"], label="Diabetes")
    ],
    outputs="text",
    title="🩺 Heart Disease Risk Checker",
    description="Enter your health information to check your heart disease risk level using a trained Logistic Regression model.",
    allow_flagging="never"
).launch()
```

---

## 📁 Folder Structure

```
📦heart-disease-risk-checker
 ┣ 📄 heart_disease_dataset.csv
 ┣ 📄 Heart_Disease_Predictor.ipynb
 ┗ 📄 README.md
```

---

## 📌 Requirements

- Python 3.8+
- pandas
- scikit-learn
- gradio

🔧 **Install dependencies**:

```bash
pip install pandas scikit-learn gradio
```

---

## 📷 Screenshot

![UI Preview](https://github.com/user-attachments/assets/b47491e7-94aa-4b2a-9a18-3f99617573e1)

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and share under the terms of the license.
