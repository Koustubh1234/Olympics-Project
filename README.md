# Olympixplore - Olympics Data Analysis Web App

## Overview
Olympixplore is an interactive web application built using **Streamlit** that provides an in-depth analysis of the Olympics dataset. The app enables users to explore historical Olympic data, analyze athlete performance, and predict medal winners using **machine learning algorithms**.

## Features
- **Dashboard Overview**: Visual insights into Olympic history, including medal trends and country-wise performance.
- **Sport Analysis**: Detailed insights into different sports, athletes, and their achievements.
- **Athlete Analysis**: Individual athlete performance analysis across different Olympics.
- **Country-wise Analysis**: Performance tracking of countries across multiple Olympic events.
- **Medal Prediction**: Using **Random Forest Algorithm**, the app predicts possible medal winners based on past data.
- **Interactive Visualizations**: Built-in support for charts and graphs using **Matplotlib, Seaborn, and Power BI**.

## Dataset
The dataset used for this project is **120 Years of Olympic History: Athletes and Results**.
- **Source**: [Kaggle](https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results)

## Tech Stack
- **Programming Language**: Python
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Random Forest Algorithm)
- **Visualization**: Matplotlib, Seaborn, Power BI
- **Deployment**: Heroku

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/olympixplore.git
   cd olympixplore
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate  # For Windows
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the application:
   ```sh
   streamlit run app.py
   ```

---
Developed with ❤️ by [koustubh sadavarte]

