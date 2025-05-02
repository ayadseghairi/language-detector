# 🌐 Language Detection using KNN

This project is a lightweight language detection tool powered by **K-Nearest Neighbors (KNN)** and **TF-IDF character-level vectorization**. It can predict the language of a single sentence interactively or process a dataset in batch mode.

---

## 📁 Project Structure

```

.
├── cleaned\_balanced\_dataset.csv     # Training dataset
├── train.py                         # Model training script
├── main.py                          # Interactive or batch prediction script
├── language\_detection\_model.pkl     # Saved trained KNN model
└── README.md                        # Project documentation

````

---

## 🚀 Features

- 🔤 Detects the language of any input sentence.
- 📂 Batch labeling support for CSV files.
- 🧠 Uses KNN (k=5) with char-level TF-IDF vectorization.
- 💾 Saves model with `joblib` for easy reuse.
- 📊 Evaluation with classification report.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ayadseghairi/language-detector.git
cd language-detector
````

### 2. Install the requirements

```bash
pip install pandas scikit-learn joblib
```

---

## 🧠 Training the Model

To train or retrain the model using your dataset:

```bash
python train.py
```

* Expects a CSV file: `cleaned_balanced_dataset.csv`
* Columns required: `Text`, `Language`
* Outputs:

  * `language_detection_model.pkl` (the trained model)
  * `test_predictions_report.csv` (optional, shows actual vs predicted)

---

## 🧪 Model Details

* **Algorithm**: K-Nearest Neighbors (KNN)
* **k**: 5
* **Vectorizer**: TF-IDF with character n-grams (1 to 3)
* **Language coverage**: Based on the training data (expandable)

---

## 🧾 Example

### 🔸 Interactive Mode

```bash
python main.py
```

Output:

```
📝 Enter a sentence (or type 'exit' to quit): Hola amigo
🔤 Predicted language: Spanish
```

### 🔸 Batch Mode (Optional)

If your `main.py` supports batch labeling (e.g. from `Evaluation_set.csv`):

```bash
# Automatically labels and saves results
python main.py
```

---

## 🧾 Sample Dataset

**cleaned\_balanced\_dataset.csv**:

```csv
Text,Language
Hello world,English
Bonjour le monde,French
Hola mundo,Spanish
```

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use and modify it.

---

## 🙌 Credits

Made with ❤️ using Python, Scikit-learn, and joblib.
Inspired by real-world language detection systems like FastText and langdetect.
