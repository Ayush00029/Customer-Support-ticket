# 📓 Notebook-by-Notebook Master Guide & Notes

A complete, detailed walkthrough of what every single `.ipynb` notebook file is doing, step-by-step.

---

## 📖 Notebook 01: `01.ipynb` — Exploratory Data Analysis (EDA)

### 1. Purpose
To inspect the raw dataset for the first time, check data cleanliness (missing values, duplicates), and analyze the target categories before writing any ML code.

### 2. Step-by-Step Code Flow
1. **Loads Data:** Reads `Dataset/all_tickets_processed_improved_v3.csv` into a Pandas DataFrame.
2. **Dimension Check:** Checks shape: **47,837 rows** and **2 columns** (`Document` and `Topic_group`).
3. **Missing Value Check:** Runs `df.isnull().sum()` $\rightarrow$ confirms **0 null values**.
4. **Duplicate Check:** Runs `df.duplicated().sum()` $\rightarrow$ confirms **0 duplicate rows**.
5. **Class Exploration:** Runs `df['Topic_group'].value_counts()` and plots bar charts to visualize category distribution.
6. **Feature Creation:** Calculates `text_length` (word count for each ticket) and saves the processed dataset as `data.csv`.

### 3. Key Finding
* **Severe Class Imbalance:**  
  * **Hardware** (13,617) and **HR Support** (10,915) make up over **51%** of the entire dataset.
  * **Administrative Rights** (1,760) is the smallest class with only **3.7%** of the data.
* **Takeaway:** You cannot use regular random splitting; you must use **stratified splitting** so minority classes aren't left out.

### 🎙️ What to Say in the Interview:
> *"In Notebook 01, I performed Exploratory Data Analysis on the 48,000 raw tickets. I verified that there were zero missing values or duplicate rows. Most importantly, I identified a significant class imbalance—Hardware and HR Support represented over half the dataset, while Administrative Rights accounted for only 3.7%. This discovery dictated that we needed stratified train-test splitting and weighted evaluation metrics."*

---

## 📖 Notebook 02: `02.ipynb` — Text Inspection & Noise Discovery

### 1. Purpose
To manually read through real tickets from different categories, analyze how employees write, and inspect conversational noise.

### 2. Step-by-Step Code Flow
1. **Loads Data:** Reads `data.csv`.
2. **Configures Display:** Sets `pd.set_option('display.max_colwidth', None)` to read full text without truncation.
3. **Filters by Category:** Filters tickets by category (e.g., `df.loc[df['Topic_group'] == 'Hardware']`) to read real examples.
4. **Observes Word Patterns:** Inspects common phrases appearing across tickets.

### 3. Key Finding
* **Heavy Conversational Noise:** Discovered that tickets frequently start with greetings (*"dear"*, *"hello"*, *"hi"*) and end with politeness words (*"thanks"*, *"regards"*, *"please"*), plus email headers (*"sent on thursday at 3pm"*).
* **Core Question Raised:** *"Do these conversational words confuse the model, and should we remove them?"*

### 🎙️ What to Say in the Interview:
> *"Notebook 02 was a qualitative text inspection scratchpad. By manually reading tickets across categories, I noticed significant conversational email noise—such as greetings, sign-offs, and email timestamps. This observation formed the hypothesis for our preprocessing experiments later in Notebook 05."*

---

## 📖 Notebook 03: `03_BaselineModel.ipynb` — The Baseline Model

### 1. Purpose
To build our first working machine learning prototype using the simplest, fastest pipeline to set a performance benchmark score that all future models must beat.

### 2. Step-by-Step Code Flow
1. **Feature Selection:** Sets `X = df['Document']` (text input) and `y = df['Topic_group']` (target labels).
2. **Train-Test Split:** Splits data **80% training (38,269)** and **20% testing (9,568)** using `stratify=y` and `random_state=42`.
3. **TF-IDF Vectorization:**
   * Runs `tfidf.fit_transform(X_train)`: Extracts **11,659 unique words** and calculates their TF-IDF scores.
   * Runs `tfidf.transform(X_test)`: Converts test tickets using that learned vocabulary without data leakage.
4. **Model Training:** Initializes `LogisticRegression(max_iter=1000, random_state=42)` and trains it with `model.fit(X_train_tfidf, y_train)`.
5. **Prediction:** Runs `y_pred = model.predict(X_test_tfidf)`.
6. **Evaluation:**
   * Calculates `accuracy_score(y_test, y_pred)` $\rightarrow$ **85.82% Accuracy**.
   * Generates `classification_report` and plots the Confusion Matrix.
7. **Export Results:** Combines `X_test`, `y_test`, and `y_pred` into a DataFrame and saves it as **`baselineResults.csv`**.

### 3. Key Finding
* Out of the box, TF-IDF + Logistic Regression achieved an impressive **85.8% accuracy**.
* Administrative Rights had the lowest recall (0.66).

### 🎙️ What to Say in the Interview:
> *"In Notebook 03, I built our baseline model using an 80:20 stratified split. I vectorized the text with TF-IDF, producing an 11,600-dimensional sparse feature space, and trained a multi-class Logistic Regression classifier. The baseline achieved 85.8% accuracy. I then exported the test predictions into `baselineResults.csv` to decouple model training from error analysis."*

---

## 📖 Notebook 04: `04_ErrorAnalysis.ipynb` — Diagnosing Mistakes

### 1. Purpose
To analyze **where, how, and why** the baseline model made mistakes.

### 2. Step-by-Step Code Flow
1. **Loads Predictions:** Reads `baselineResults.csv` (contains 9,568 test tickets).
2. **Filters Correct vs. Wrong:**
   * `correct_predictions = results[results['Actual'] == results['Predicted']]` $\rightarrow$ **8,211 correct**.
   * `wrong_predictions = results[results['Actual'] != results['Predicted']]` $\rightarrow$ **1,357 mistakes**.
3. **Calculates Error Rate:** Computes $\frac{1357}{9568} \times 100 = \mathbf{14.18\%}$ misclassification rate.
4. **Manual Error Inspection:** Inspects the misclassified tickets line-by-line.

### 3. Key Findings
1. **Semantic Overlap:** The model struggled between *Administrative Rights* and *Access* because both use words like *"permission"*, *"rights"*, and *"login"*.
2. **Email Noise:** Many misclassified tickets had heavy greetings (*"hi"*, *"thanks"*).
3. **Ground-Truth Label Noise:** Discovered that some "errors" were actually due to human employees selecting the wrong dropdown category when submitting the ticket (e.g., Row 43906).

### 🎙️ What to Say in the Interview:
> *"In Notebook 04, I performed targeted Error Analysis by isolating the 1,357 misclassified test tickets—a 14.2% error rate. By inspecting these failures, I discovered that the minority class 'Administrative Rights' frequently overlapped with 'Access' due to shared vocabulary around user permissions. I also identified real-world ground-truth label noise in the historical data."*

---

## 📖 Notebook 05: `05_TextPreprocessing.ipynb` — Controlled Cleaning Experiments

### 1. Purpose
To scientifically test: *"Will removing punctuation, numbers, and custom email stopwords improve our 85.8% baseline?"*

### 2. Step-by-Step Code Flow
1. **Builds Cleaning Function (`preprocess_text`):**
   * Lowercase conversion.
   * Regex removal of numbers (`\d+`) and punctuation (`[^a-zA-Z\s]`).
   * Strips extra whitespace.
   * Removes 179 standard English stopwords + custom email stopwords (*"dear"*, *"hello"*, *"thanks"*, *"regards"*, days of week, months).
2. **Applies Cleaning:** Creates `df['Clean_Document']`.
3. **Trains & Evaluates:**
   * Uses the exact same split (`test_size=0.2, random_state=42, stratify=y`).
   * Transforms with TF-IDF and retrains Logistic Regression.
   * Calculates `accuracy_score`.

### 3. Key Finding
* **Accuracy dropped from 85.82% to 85.08%!**
* **Reason:** TF-IDF’s Inverse Document Frequency was already automatically penalizing common email greetings. Manual cleaning accidentally deleted helpful technical tokens and numbers.
* **Engineering Decision:** Reject the cleaning pipeline and retain the original raw text.

### 🎙️ What to Say in the Interview:
> *"In Notebook 05, I ran a controlled experiment to test if text cleaning and custom email stopword removal would improve performance. Accuracy actually dropped from 85.82% to 85.08%. This proved empirically that TF-IDF's inverse document frequency already downweighted conversational noise automatically. We therefore decided to keep the raw text, avoiding unnecessary pipeline complexity."*

---

## 📖 Notebook 06: `06_ModelComparison.ipynb` — Algorithm Benchmarking

### 1. Purpose
To objectively compare multiple machine learning algorithms on the exact same TF-IDF features and pick the clear winner.

### 2. Step-by-Step Code Flow
1. **Defines 4 Models:**
   * `LogisticRegression(max_iter=1000)`
   * `LinearSVC()` (Linear SVM)
   * `RandomForestClassifier(random_state=42)`
   * `MultinomialNB()` (Naive Bayes)
2. **Runs Benchmark Loop:**
   * Loops through each model, trains it on `X_train_tfidf`, and evaluates it on `X_test_tfidf`.
   * Records Accuracy, Precision, Recall, and Weighted F1-score into a comparison DataFrame.

### 3. Benchmark Results
| Algorithm | Accuracy | Weighted F1 | Why it performed this way |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | **85.82%** | **85.83%** 🥇 | **Winner:** Cleanly separates high-dimensional sparse text vectors; calibrated probabilities. |
| **Linear SVM** | 85.61% | 85.62% 🥈 | **Close 2nd:** Excellent linear boundary, but lacks native probability outputs. |
| **Random Forest** | 83.09% | 83.05% 🥉 | Decision trees struggle with sparse 11,000-word orthogonal feature spaces. |
| **Multinomial Naive Bayes** | 73.76% | 72.60% 4th | Fails because word independence assumption is violated in IT tickets. |

### 🎙️ What to Say in the Interview:
> *"In Notebook 06, I benchmarked four classifiers across identical TF-IDF features. Logistic Regression achieved the highest weighted F1 score at 85.83%, followed closely by Linear SVM at 85.62%. Random Forest and Naive Bayes underperformed due to the high-dimensional sparse nature of the text data. Logistic Regression was selected as our production algorithm because it offered the highest accuracy, fast CPU inference, and calibrated probability outputs."*

---

## 📖 Notebook 07: `07_FinalModel.ipynb` — Production Model Export

### 1. Purpose
To train the final production pipeline and serialize the model artifacts to disk so the Streamlit web application can load them.

### 2. Step-by-Step Code Flow
1. **Loads Data:** Reads the full dataset.
2. **Splits Data:** Uses the established 80% training split.
3. **Fits Vectorizer:** Fits `TfidfVectorizer` on the text.
4. **Trains Model:** Trains `LogisticRegression(max_iter=1000)`.
5. **Serializes Artifacts (Joblib):**
   * Saves `tfidf` as `Models/tfidf_vectorizer.pkl`.
   * Saves `model` as `Models/logistic_model.pkl`.

### 3. Output
* Creates the two production `.pkl` files that power the live interactive web app (`app.py`).

### 🎙️ What to Say in the Interview:
> *"In Notebook 07, I trained the final production pipeline and serialized the fitted TF-IDF vectorizer and Logistic Regression classifier into `.pkl` files using Joblib. These serialized artifacts are loaded by our Streamlit web application in `app.py`, enabling instant sub-5-millisecond live inference for end users."*
