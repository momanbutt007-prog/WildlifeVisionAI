
# 🦓 Wildlife Vision AI

A polished wildlife image classification application powered by TensorFlow and a trained EfficientNetV2-S model.

## Recognized Classes

- Buffalo
- Elephant
- Rhino
- Zebra

## Final Validation Performance

| Metric | Result |
|---|---:|
| Accuracy | 98.01% |
| Precision | 98.11% |
| Recall | 98.01% |
| F1 Score | 98.02% |
| Validation Loss | 0.18928 |

## Project Structure

```text
Wildlife_Classifier/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── render.yaml
├── runtime.txt
├── .python-version
├── .gitignore
├── README.md
│
├── models/
│   ├── wildlife_classifier_final.keras
│   ├── class_names.json
│   └── README.txt
│
├── templates/
│   ├── index.html
│   └── about.html
│
└── static/
    ├── css/
    │   └── style.css
    ├── js/
    │   └── app.js
    ├── uploads/
    └── results/
```

# 1. Add the trained model

Copy your trained model from Colab into:

```text
models/wildlife_classifier_final.keras
```

The included `class_names.json` contains:

```json
[
  "buffalo",
  "elephant",
  "rhino",
  "zebra"
]
```

The numeric order MUST remain the same as the order used during training.

# 2. Python 3.12 virtual environment

Windows:

```powershell
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# 3. Run Flask

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

# 4. Run Streamlit

```powershell
streamlit run streamlit_app.py
```

# 5. GitHub

```powershell
git init
git add .
git commit -m "Initial wildlife AI classifier"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

# 6. Deploy Flask on Render

Create a new Render Web Service connected to the GitHub repository.

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
gunicorn app:app --workers 1 --threads 2 --timeout 120
```

Python version:

```text
3.12.0
```

The repository already contains `render.yaml`, `runtime.txt`, and `.python-version`.

## Important Render note

TensorFlow models can make deployments large and slow. If GitHub rejects the `.keras` file because of its size, use Git LFS or external model storage rather than committing a very large binary normally.

# 7. Streamlit deployment

For Streamlit Community Cloud, use:

```text
streamlit_app.py
```

and make sure the `models/` folder and model file are available to the deployed application.

# 8. Model input

The application resizes uploaded images to:

```text
384 × 384
```

before inference, matching the training configuration.

# 9. Render health check

The Flask application provides:

```text
/health
```

Example response:

```json
{
  "status": "healthy",
  "model_available": true,
  "classes": [
    "buffalo",
    "elephant",
    "rhino",
    "zebra"
  ]
}
```

## Disclaimer

This is an educational/computer-vision classification project. Predictions are model outputs and should not be treated as expert wildlife identification.
