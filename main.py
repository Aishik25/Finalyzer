import pandas as pd
import spacy
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import os
from fuzzywuzzy import fuzz, process
import logging
from pathlib import Path
import yaml
import PyPDF2
import pytesseract
import streamlit as st
import tempfile
from pdf2image import convert_from_ path
import plotly.express as px
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finance_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config() -> Dict:
    config_path = Path("config.yaml")
    default_config = {
        'budget_limits': {
            'food': 3000,
            'transport': 1500,
            'subscriptions': 1500,
            'shopping': 2000,
            'utilities': 2000,
            'financial_services': 2000,
            'miscellaneous': 1500
        },
        'category_keywords': {
            'food': ['zomato', 'swiggy', 'restaurant', 'dine', 'cafe'],
            'transport': ['uber', 'ola', 'lyft', 'bus', 'train', 'metro'],
            'subscriptions': ['netflix', 'spotify', 'prime', 'disney+'],
            'shopping': ['amazon', 'flipkart', 'myntra', 'ajio'],
            'utilities': ['electricity', 'water', 'gas', 'wifi', 'internet']
        },
        'anomaly_weights': {
            'shopping': 0.15,
            'food': 0.1,
            'subscriptions': 0.05,
            'transport': 0.1,
            'utilities': 0.02,
            'financial_services': 0.05,
            'miscellaneous': 0.1
        }
    }
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                for section in default_config:
                    if section in user_config:
                        default_config[section].update(user_config[section])
                return default_config
        except Exception as e:
            logger.warning(f"Config load error: {e}. Using defaults")
    return default_config

config = load_config()

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])

def clear_previous_data():
    """Clean up previous analysis files"""
    files = ["analyzed_statement.csv", "expense_chart.html"]
    for filename in files:
        try:
            if os.path.exists(filename):
                os.remove(filename)
                logger.info(f"Cleared previous file: {filename}")
        except Exception as e:
            logger.error(f"Error deleting {filename}: {e}")

def load_statement(file) -> Optional[pd.DataFrame]:
    try:
        if file.name.lower().endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.lower().endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            text = ""
            with open(tmp_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text += page.extract_text()
            
            if len(text.strip()) < 100:
                images = convert_from_path(tmp_path)
                text = "\n".join(pytesseract.image_to_string(img) for img in images)

            lines = [line.split(',') for line in text.split('\n') if line.strip() and ',' in line]
            df = pd.DataFrame(lines, columns=["Date", "Description", "Amount"])
            
            os.unlink(tmp_path) 
        else:
            raise ValueError("Unsupported file format. Please upload CSV or PDF.")
        
        required_cols = {'Date', 'Description', 'Amount'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        df['Amount'] = pd.to_numeric(
            df['Amount'].astype(str).str.replace('[^\d.-]', '', regex=True),
            errors='coerce'
        )
        df = df.dropna(subset=['Amount'])
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=['Date'])
        
        logger.info(f"Loaded {len(df)} valid transactions")
        return df
    
    except Exception as e:
        logger.error(f"Error loading statement: {e}")
        st.error(f"❌ File processing failed: {str(e)}")
        return None

def classify_transaction(description: str) -> str:
    """Categorize transactions using multi-stage matching"""
    if not isinstance(description, str):
        return 'unclassified'
    
    desc = description.lower()

    for category, keywords in config['category_keywords'].items():
        if any(keyword in desc for keyword in keywords):
            return category

    best_match, score = process.extractOne(
        desc, 
        config['category_keywords'].keys(), 
        scorer=fuzz.token_set_ratio
    )
    if score > 70:
        return best_match

    doc = nlp(description)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT'] and 'bank' not in ent.text.lower():
            return 'financial_services'
    
    return 'miscellaneous'

def detect_anomalies(df: pd.DataFrame) -> List[str]:
    try:
        results = []
        for category, sub_df in df.groupby('Category'):
            if len(sub_df) < 5:
                results.extend(['Normal'] * len(sub_df))
                continue
                
            model = IsolationForest(
                contamination=config['anomaly_weights'].get(category, 0.1),
                random_state=42
            )
            amounts = sub_df['Amount'].values.reshape(-1, 1)
            preds = model.fit_predict(amounts)
            results.extend(['High Risk' if x == -1 else 'Normal' for x in preds])
        
        return results
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        return ['Error'] * len(df)

def suggest_savings(category_totals: Dict[str, float], income: float = None) -> List[str]:
    suggestions = []
    for category, spent in category_totals.items():
        budget = config['budget_limits'].get(category, float('inf'))
        overspend = abs(spent) - budget
        
        if overspend > 0:
            tips = {
                'food': [
                    f"Meal prep could save ~₹{min(2000, overspend//2)}",
                    "Try grocery shopping instead of dining out"
                ],
                'shopping': [
                    "Implement 48-hour waiting period",
                    f"Cancel 1 subscription to save ~₹{overspend//3}"
                ],
                'subscriptions': [
                    "Bundle services for discounts",
                    "Cancel unused memberships"
                ],
                'transport': [
                    "Use public transit 2x/week",
                    "Carpool to save on fuel"
                ]
            }
            tip = tips.get(category, ["Review spending habits"])[0]
            suggestions.append(
                f"⚠️ {category.title()}: Overspent ₹{overspend:.2f} "
                f"(Budget ₹{budget}). Tip: {tip}"
            )
        else:
            suggestions.append(
                f"✅ {category.title()}: ₹{abs(spent):.2f}/₹{budget} "
                f"(₹{budget - abs(spent):.2f} remaining)"
            )
    
    if income and income > 0:
        total_spent = sum(abs(x) for x in category_totals.values())
        savings_rate = (income - total_spent) / income * 100
        suggestions.append(
            f"\n💡 Monthly Summary: Spent ₹{total_spent:.2f} of ₹{income:.2f} "
            f"({savings_rate:.1f}% savings rate)"
        )
    
    return suggestions

def plot_expenses(category_totals: Dict[str, float]):
    df = pd.DataFrame({
        'Category': category_totals.keys(),
        'Amount': [abs(x) for x in category_totals.values()]
    })
    threshold = 0.05 * df['Amount'].sum()
    small_cats = df[df['Amount'] < threshold]
    if len(small_cats) > 0:
        other_row = pd.DataFrame({
            'Category': ['Other'],
            'Amount': [small_cats['Amount'].sum()]
        })
        df = pd.concat([df[df['Amount'] >= threshold], other_row])
    
    fig = px.pie(
        df,
        values='Amount',
        names='Category',
        title='Monthly Expense Distribution',
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>₹%{value:.2f} (%{percent})"
    )
    fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        margin=dict(t=50, b=20, l=20, r=20)
    )
    fig.write_html("expense_chart.html")
    return fig

def main():
    st.set_page_config(
        page_title="AI Finance Analyzer",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .header-style { font-size:24px; font-weight:bold; color:#2E86C1; }
    .warning-text { color: #E74C3C; }
    .success-text { color: #27AE60; }
    .summary-text { font-weight:bold; color:#3498DB; }
    .stButton>button { background-color: #2E86C1; color:white; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="header-style">💰 AI-Powered Personal Finance Analyzer</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader(
            "Choose bank statement (CSV/PDF)",
            type=['csv', 'pdf'],
            help="File should contain Date, Description, Amount columns"
        )
        monthly_income = st.number_input(
            "Monthly Income (₹):",
            min_value=0,
            value=50000,
            step=1000
        )
        st.markdown("---")
        st.markdown("**Tip:** For PDF statements, ensure text is selectable for best results")
    
    if uploaded_file is not None:
        with st.spinner("Analyzing your transactions..."):
            df = load_statement(uploaded_file)
            
        if df is not None and not df.empty:
            st.success(f"✅ Processed {len(df)} transactions successfully")

            df['Category'] = df['Description'].apply(classify_transaction)
            df['Risk'] = detect_anomalies(df)

            csv_path = "analyzed_transactions.csv"
            df.to_csv(csv_path, index=False)

            category_totals = df.groupby('Category')['Amount'].sum().to_dict()
            insights = suggest_savings(category_totals, monthly_income)

            st.subheader("📊 Spending Insights")
            for insight in insights:
                if insight.startswith("⚠️"):
                    st.markdown(f'<p class="warning-text">{insight}</p>', unsafe_allow_html=True)
                elif insight.startswith("✅"):
                    st.markdown(f'<p class="success-text">{insight}</p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="summary-text">{insight}</p>', unsafe_allow_html=True)

            st.subheader("📈 Expense Breakdown")
            fig = plot_expenses(category_totals)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                with open(csv_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Analysis (CSV)",
                        data=f,
                        file_name="financial_analysis.csv",
                        mime="text/csv"
                    )
            with col2:
                with open("expense_chart.html", "rb") as f:
                    st.download_button(
                        label="📊 Download Chart (HTML)",
                        data=f,
                        file_name="expense_chart.html",
                        mime="text/html"
                    )

            if st.checkbox("Show transaction details"):
                st.dataframe(
                    df.sort_values('Amount', ascending=False),
                    height=400,
                    use_container_width=True
                )
        else:
            st.error("No valid transactions found in the uploaded file.")

if __name__ == "__main__":
    clear_previous_data()
    main()