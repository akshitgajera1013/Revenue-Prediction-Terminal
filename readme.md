# 🛒 E-Commerce Revenue Prediction Engine (Enterprise Edition)

Deployment Link :- https://revenue-prediction-terminal.streamlit.app/

About Dataset :-
This dataset contains session-level information about users visiting an online shopping website. The goal of the dataset is to predict whether a visitor will generate revenue (make a purchase) or not during a browsing session.

The dataset includes various behavioral and technical attributes such as the number of pages visited, time spent on pages, bounce rates, traffic sources, visitor type, and other session metrics.

It is commonly used for classification problems in machine learning, particularly for predicting customer purchase intent.

Dataset Overview

     Property	Value
     Total Records	12,330
     Total Features	18
     Target Variable	Revenue
     Task Type	Binary Classification
     Target Variable

Revenue

 This column indicates whether the user completed a purchase.

           Value	Meaning
           False	No purchase made
           True	Purchase completed

Distribution:

      False (No Purchase) : 10,422 sessions
      True (Purchase) : 1,908 sessions
      
This indicates the dataset is imbalanced, with fewer purchase events.

Feature Description

      Page Interaction Features
      Feature	Description
      Administrative	Number of administrative pages visited
      Administrative_Duration	Time spent on administrative pages
      Informational	Number of informational pages visited
      Informational_Duration	Time spent on informational pages
      ProductRelated	Number of product-related pages visited
      ProductRelated_Duration	Time spent on product pages
 
Website Behavior Metrics

 Feature	Description
 
      BounceRates	Percentage of visitors leaving after viewing one page
      ExitRates	Percentage of exits from the website
      PageValues	Average value of pages visited before purchase
      SpecialDay	Indicates closeness to special days (like holidays)
 
Visitor and Technical Information
     
      Feature	Description
      Month	Month of the visit
      OperatingSystems	Operating system used by the visitor
      Browser	Browser used
      Region	Geographic region
      TrafficType	Traffic source type
 
Visitor Characteristics

      Feature	Description
      VisitorType	Type of visitor (Returning, New, Other)
      
 Weekend	Whether the session occurred on weekend

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Random Forest](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Data%20Viz-Plotly-purple?style=for-the-badge)


An advanced, monolithic Python application built for digital retail analytics and shopper intent forecasting. This system leverages a highly optimized **Random Forest Classifier** to process 13 distinct website session vectors, accurately predicting whether a user's browsing session will culminate in a financial transaction (Revenue). 

## 🧠 System Architecture & Capabilities

This platform processes raw digital telemetry into actionable commercial intelligence, actively accounting for the extreme class imbalance typically found in digital retail datasets.

### 1. Shopper Intent Classification
Users input critical session data across three behavioral domains: Session Durations (Administrative, Informational, Product-Related), Engagement Metrics (Exit Rates, Page Values, Seasonal Proximity), and Technical Profiles (Browser, OS, Traffic Type). The Python-based UI processes these vectors through the Random Forest inference kernel to generate an immediate purchasing probability.

### 2. Session Analytics & Radar Mapping
Generates a multi-dimensional radar topology of the active session, comparing normalized engagement metrics against simulated global baseline behaviors. Includes an interactive probability distribution mapping the exact likelihood of a revenue event.

### 3. Asymmetric Class Handling (Model Evaluation)
E-commerce conversion datasets are notoriously imbalanced (the vast majority of sessions result in abandonment). This model was engineered specifically to optimize beyond standard accuracy, focusing on identifying true buyers without triggering excessive false positives:
* **Accuracy:** `90.70%` (Baseline performance metric)
* **Precision (Class 1):** `74.19%` (High confidence when the model explicitly predicts a purchase)
* **Recall (Class 1):** `57.18%` (Successfully captures the majority of actual buyers within the imbalanced data)
* **F1 Score:** `64.59%` (The harmonic mean, providing a balanced view of model capability)

### 4. Revenue Impact Simulator
Simulates a predictive conversion uplift trajectory. By manipulating high-weight features (such as `PageValues`), the system forecasts how improvements in User Experience (UX) or targeted product placement dynamically increase the mathematical probability of checkout completion.

### 5. Behavioral Variance (Monte Carlo Simulation)
Executes a 100-iteration stochastic mathematical simulation to model cohort session volatility. Applies the model's inherent error variance to map the unpredictability of human purchasing behavior (e.g., unexpected cart abandonment, payment friction, sudden session bounces).

### 6. Secure Data Export (JSON / CSV)
Generates an official Purchase Dossier tagged with a unique cryptographic Session ID. Enables base64-encoded, secure local downloads of the entire inference payload in both programmatic (JSON) and ledger (CSV) formats for database integration.


## 🛠️ Technical Stack

* **Core Logic & Computation:** 
* **Data Processing & Pipelines:** `pandas`
* **Machine Learning Architecture:** `scikit-learn` (Random Forest Classifier, Label Encoding)
* **Interactive Data Visualization:** `plotly.express`, `plotly.graph_objects`
* **Frontend Delivery:** Custom Python-rendered UI engine utilizing over 350 lines of injected, dynamic CSS (Glassmorphism, Keyframe Data-Packet Animations, Responsive Flexboxes).

---

## 📂 Repository Structure

├── app.py

├── model.pkl   

├── encoder.pkl  

├── requirements.txt   

└── README.md               


⚙️ Installation & Deployment

Clone the Repository

Move 

Install Dependencies

Run App
 
     git clone https://github.com/akshitgajera1013/Revenue-Prediction-Terminal.git
     cd ecommerce-revenue-prediction
     pip install -r requirements.txt
     Initialize the Application Server
     python -m streamlit run app.py or streamlit run app.py


⚠️ Data Privacy Disclaimer
Strictly Confidential Commercial Data. This intelligence terminal and its underlying Random Forest architecture are designed for educational, data science, and theoretical forecasting purposes only. The outputs generated by the simulations are probabilistic forecasts based on historical telemetry and do not guarantee future commercial revenue.
