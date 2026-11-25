# 📝 Complete Interview Revision Notes
## Customer Support Ticket Classification using NLP

---

### 1. 🎯 Project Overview (The Big Picture)
* **Problem:** Large companies receive thousands of IT tickets daily via email. Manual reading and sorting by human agents is slow, expensive, and error-prone.
* **Solution:** An end-to-end NLP system that automatically classifies tickets into **8 departments** in **< 5 milliseconds**.
* **Tech Stack:** Python, Pandas, Scikit-learn (TF-IDF + Logistic Regression), Joblib, Streamlit.
* **Accuracy / F1:** **85.8%** Weighted F1-Score.

---

### 2. 📊 Dataset Summary
* **Source:** Enterprise IT Support Tickets (`all_tickets_processed_improved_v3.csv`).
* **Volume:** **47,837 tickets**, 2 columns: `Document` (text) and `Topic_group` (label).
* **8 Classes:**
  1. **Hardware** (13,617 tickets — 28.5%)
  2. **HR Support** (10,915 tickets — 22.8%)
  3. **Access** (7,125 tickets — 14.9%)
  4. **Miscellaneous** (7,060 tickets — 14.8%)
  5. **Storage** (2,777 tickets — 5.8%)
  6. **Purchase** (2,464 tickets — 5.2%)
  7. **Internal Project** (2,119 tickets — 4.4%)
  8. **Administrative Rights** (1,760 tickets — 3.7%) — *Minority class*
* **Imbalance Note:** Hardware & HR make up >51% of the data. We used a **stratified split** (`stratify=y`) and evaluated using **Weighted F1** to account for this.

---

### 3. ⚙️ How It Works (The 3-Step Pipeline)
$$\text{Raw Text (Email)} \xrightarrow{\quad\text{TF-IDF}\quad} \text{Numeric Feature Vector} \xrightarrow{\quad\text{Logistic Regression}\quad} \text{Predicted Department}$$

1. **Input:** User types problem in Streamlit (`app.py`).
2. **Feature Extraction (TF-IDF):** Converts text into an 11,659-number vector. Downweights common words (*"dear"*, *"thanks"*) and boosts keywords (*"password"*, *"printer"*).
3. **Classification (Logistic Regression):** Multiplies numbers by learned weights, computes Softmax probabilities across 8 classes, and outputs the top category.

---

### 4. 📓 The 7 Notebooks (The Machine Learning Lifecycle)
* **`01.ipynb` (EDA):** Checked data hygiene (0 nulls, 0 duplicates) and graphed class imbalance.
* **`02.ipynb` (Text Inspection):** Read sample tickets; observed conversational email noise.
* **`03_BaselineModel.ipynb` (Baseline):** 80:20 stratified split + TF-IDF + Logistic Regression = **85.8% accuracy**. Exported predictions to `baselineResults.csv`.
* **`04_ErrorAnalysis.ipynb` (Diagnostics):** Isolated the 1,357 misclassified tickets (14.2% error rate). Found Administrative Rights had the lowest recall (0.66).
* **`05_TextPreprocessing.ipynb` (Experiment):** Tested removing punctuation, numbers, and email stopwords. Accuracy **dropped to 85.08%**! Proved TF-IDF already filters noise mathematically. *Decision: Keep raw text.*
* **`06_ModelComparison.ipynb` (Benchmarking):** Compared 4 models on identical TF-IDF features:
  * **Logistic Regression:** **85.83% F1** (Winner)
  * **Linear SVM:** 85.62% F1
  * **Random Forest:** 83.05% F1
  * **Naive Bayes:** 72.60% F1
* **`07_FinalModel.ipynb` (Export):** Saved `logistic_model.pkl` and `tfidf_vectorizer.pkl` via Joblib for the Streamlit web app.

---

### 5. 🌟 3 Killer Stories for the Interview (Shows Real Senior Thinking)

#### Story 1: Why We Dropped Stopword Removal
* We tested removing email noise (*"dear"*, *"thanks"*).
* Accuracy dropped from 85.8% to 85.0%.
* **Reason:** TF-IDF’s Inverse Document Frequency ($\text{IDF}$) mathematically downweights ubiquitous words automatically, while regex cleaning accidentally deleted helpful technical tokens.

#### Story 2: The "VPN" Out-Of-Vocabulary (OOV) Edge Case
* Query: *"Unable to connect to corporate VPN"* predicted **Hardware** (90%) instead of **Access**!
* **Root cause:** The term `"VPN"` appeared 0 times in the 48k training tickets. TF-IDF dropped it as OOV. The model only saw `"connect"` and `"network"`, which in this dataset are weighted heavily toward physical Hardware (cables/routers).
* **Takeaway:** Demonstrates why dense embeddings/subwords (like BERT) handle unseen jargon better than exact-match TF-IDF.

#### Story 3: Ground-Truth Label Noise (Row 43906)
* Ticket text: *"trying run below report seems have access could you please check why should have access reports"*
* Ground-truth label in dataset: **Hardware**!
* **Reason:** In real-world enterprise IT, users frequently pick the default dropdown category (often Hardware) when submitting tickets, creating human label noise in historical data.

---

### 6. 🧠 Essential Definitions in Plain English
* **TF-IDF:** Turns words into numbers. Ignores common words and highlights rare keywords.
* **Train-Test Split:** 80% to teach the model (train), 20% to test on unseen data (test).
* **Stratify (`stratify=y`):** Keeps the percentage of each department equal in both train and test sets.
* **Logistic Regression:** A linear classifier that calculates probability percentages across all categories using a Softmax function.
* **F1-Score:** The harmonic mean of Precision (avoiding false alarms) and Recall (avoiding missed tickets).
* **`.pkl` File (Joblib):** A "save game" file that saves the trained model so you don't have to retrain every time.
* **Streamlit:** Python framework to build the web interface.

---

### 7. ⏱️ The 60-Second Interview Pitch (Memorize This!)

> *"For this project, I built an automated IT support ticket classification system trained on 48,000 real-world tickets across 8 department categories.*
> 
> *Using an 80:20 stratified split, I established a baseline model using TF-IDF and Logistic Regression, achieving an 85.8% weighted F1-score.*
> 
> *During error analysis, I tested whether cleaning conversational email greetings like 'dear' and 'thanks' would improve accuracy. Surprisingly, accuracy dropped to 85.0%, proving that TF-IDF's Inverse Document Frequency was already filtering out email noise automatically. I therefore retained the raw text pipeline.*
> 
> *After benchmarking against Linear SVM, Random Forest, and Naive Bayes, Logistic Regression delivered the best balance of accuracy and sub-5ms CPU latency. Finally, I serialized the model and vectorizer with Joblib and deployed an interactive Streamlit web application."*
