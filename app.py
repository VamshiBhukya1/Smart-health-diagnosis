# import streamlit as st
# # Must be the first Streamlit command
# st.set_page_config(page_title="AI Health Assistant", layout="wide")

# import pandas as pd
# import os
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# # ✅ Data folder (same directory as app.py)
# DATA_DIR = os.path.join(os.path.dirname(__file__), 'New folder')

# # ✅ Check CSVs exist
# def check_files():
#     required = ['dataset.csv', 'symptom_Description.csv', 'Symptom-severity.csv', 'symptom_precaution.csv']
#     missing = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]
#     if missing:
#         st.error(f"Missing files: {', '.join(missing)}. Place all CSVs in same folder as app.py.")
#         st.stop()

# @st.cache_data
# def load_data():
#     check_files()
#     train = pd.read_csv(os.path.join(DATA_DIR, 'dataset.csv'))
#     desc = pd.read_csv(os.path.join(DATA_DIR, 'symptom_Description.csv'))
#     sev = pd.read_csv(os.path.join(DATA_DIR, 'Symptom-severity.csv'))
#     prec = pd.read_csv(os.path.join(DATA_DIR, 'symptom_precaution.csv'))
#     return train, desc, sev, prec

# train_df, desc_df, sev_df, prec_df = load_data()

# @st.cache_data
# def train_model():
#     # Get all unique symptoms from the symptom columns
#     symptom_columns = [col for col in train_df.columns if col.startswith('Symptom_')]
    
#     # Clean and collect all unique symptoms
#     all_symptoms = set()
#     for col in symptom_columns:
#         # Clean symptom names: lowercase, remove extra spaces, replace spaces with underscore
#         symptoms = train_df[col].dropna().apply(lambda x: x.strip().lower().replace(' ', '_'))
#         all_symptoms.update(symptoms.unique())
#     all_symptoms = sorted(list(all_symptoms))
    
#     # Create feature matrix with binary encoding for symptoms
#     X = pd.DataFrame(0, index=train_df.index, columns=all_symptoms)
    
#     # Fill in symptoms presence (1) for each row
#     for col in symptom_columns:
#         for idx, symptom in train_df[col].items():
#             if pd.notna(symptom):
#                 # Clean symptom name same way as above
#                 clean_symptom = symptom.strip().lower().replace(' ', '_')
#                 if clean_symptom in all_symptoms:  # ensure symptom exists in our columns
#                     X.loc[idx, clean_symptom] = 1
    
#     # Verify no NaN values exist
#     X = X.fillna(0)  # explicitly fill any remaining NaN with 0
    
#     # Prepare target variable
#     y = train_df['Disease']
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)
    
#     # Train model
#     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
    
#     return model, le, list(X.columns)

# model, le, symptom_cols = train_model()
# # create a lowercase index mapping for symptom matching in the UI
# symptom_cols_lower = [c.lower() for c in symptom_cols]

# # Helper function to clean symptom names
# def clean_symptom_name(symptom):
#     return symptom.strip().lower().replace(' ', '_')

# # ✅ Streamlit setup
# st.title("🤖 AI Health Assistant")

# if 'page' not in st.session_state:
#     st.session_state.page = "Home"

# # Sidebar navigation
# page = st.sidebar.selectbox("Navigate", ["Home", "Symptoms", "Results"], index=["Home", "Symptoms", "Results"].index(st.session_state.page))

# # ✅ Home Page
# if page == "Home":
#     st.header("Welcome to AI Health Assistant")
#     st.write("Get AI-based predictions for possible diseases based on your symptoms.")
#     if st.button("🚀 Start Assessment"):
#         st.session_state.page = "Symptoms"
#         st.rerun()

# # ✅ Symptoms Page
# elif page == "Symptoms":
#     st.header("🔍 Select Your Symptoms")
#     search = st.text_input("Search symptoms...")
#     # Display symptoms in a more readable format (replace underscores with spaces)
#     display_symptoms = [s.replace('_', ' ').title() for s in symptom_cols]
#     search_lower = search.lower()
#     filtered_indices = [i for i, s in enumerate(display_symptoms) if search_lower in s.lower()] if search else range(len(display_symptoms))
#     filtered_display = [display_symptoms[i] for i in filtered_indices]
#     selected_display = st.multiselect("Choose symptoms", filtered_display)
#     # Convert selected symptoms back to internal format
#     selected = [clean_symptom_name(s) for s in selected_display]

#     if st.button("📊 Predict Diseases"):
#         if not selected:
#             st.warning("Please select at least one symptom.")
#         else:
#             # Feature vector
#             input_vec = np.zeros(len(symptom_cols))
#             for s in selected:
#                 if s in symptom_cols:  # symptom_cols already contains cleaned names
#                     idx = symptom_cols.index(s)
#                     input_vec[idx] = 1

#             # Predictions for all diseases
#             probs = model.predict_proba([input_vec])[0]
#             sorted_idx = np.argsort(probs)[::-1]
#             diseases = le.inverse_transform(sorted_idx)
#             probs_sorted = probs[sorted_idx] * 100

#             # Risk level calculation based on symptom severity
#             # Convert selected symptoms back to the format in severity CSV
#             clean_selected = [s.replace('_', ' ') for s in selected]
#             weights = sev_df[sev_df['Symptom'].str.lower().isin(clean_selected)]['weight'].mean()
#             weights = 1.0 if pd.isna(weights) else weights
#             if weights < 1.5:
#                 risk, emoji = "Low", "🟢"
#             elif weights < 2.5:
#                 risk, emoji = "Medium", "🟡"
#             else:
#                 risk, emoji = "High", "🔴"

#             # Get top disease and its information
#             top_disease = diseases[0]  # Get the most likely disease
            
#             # Get disease description
#             disease_desc = desc_df[desc_df['Disease'] == top_disease]['Description'].iloc[0] if len(desc_df[desc_df['Disease'] == top_disease]) > 0 else ""
            
#             # Get precautions for the predicted disease
#             precautions = set()
#             if len(diseases) > 0:
#                 disease_precautions = prec_df[prec_df['Disease'] == top_disease]
#                 if len(disease_precautions) > 0:
#                     for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
#                         if col in disease_precautions.columns and pd.notna(disease_precautions[col].iloc[0]):
#                             precautions.add(disease_precautions[col].iloc[0])

#             # Store results
#             st.session_state.diseases = diseases
#             st.session_state.probs = probs_sorted
#             st.session_state.risk = risk
#             st.session_state.emoji = emoji
#             st.session_state.selected = selected
#             st.session_state.precautions = list(precautions)
#             st.success("Prediction complete! Check the Results tab.")
#             st.session_state.page = "Results"
#             st.rerun()

# # ✅ Results Page
# elif page == "Results":
#     st.header("📋 Your Health Assessment Results")
#     st.info("These predictions are for educational purposes. Always consult a medical professional.")

#     if 'diseases' in st.session_state:
#         diseases = st.session_state.diseases
#         probs = st.session_state.probs
#         risk = st.session_state.risk
#         emoji = st.session_state.emoji
#         selected = st.session_state.selected
#         precautions = st.session_state.precautions

#         # Print/Export Buttons
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("🖨️ Print"):
#                 st.info("Use Ctrl+P to print.")
#         with col2:
#             if st.button("📥 Export PDF"):
#                 st.info("Use browser 'Save as PDF' to export.")

#         # Display all diseases
#         st.subheader("Prediction Results (All Possible Conditions)")
#         for i, (dis, prob) in enumerate(zip(diseases, probs), 1):
#             color = "green" if prob < 30 else "yellow" if prob < 60 else "red"
#             st.markdown(f"### {i}. {dis} {emoji} ({risk} Risk)")
#             st.progress(prob / 100)
#             st.write(f"**Match Probability:** {prob:.1f}%")
#             st.markdown("---")

#         # Symptoms
#         st.markdown("### 🧠 Matched Symptoms")
#         st.write(", ".join([s.title() for s in selected]))

#         # Precautions
#         if precautions:
#             st.markdown("### ✅ Recommended Precautions")
#             for p in precautions:
#                 st.write(f"- {p}")
#         else:
#             st.info("No specific precautions found. Please consult a doctor.")
#     else:
#         st.info("Please complete a prediction first in the Symptoms tab.")

# # Footer
# st.markdown("---")
# st.caption("⚕️ Powered by Streamlit & Scikit-learn | For educational use only.")

import streamlit as st
# Must be the first Streamlit command
st.set_page_config(page_title="AI Health Assistant", layout="wide")

import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ✅ Data folder (same directory as app.py)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'New folder')

# ✅ Check CSVs exist
def check_files():
    required = ['dataset.csv', 'symptom_Description.csv', 'Symptom-severity.csv', 'symptom_precaution.csv']
    missing = [f for f in required if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        st.error(f"Missing files: {', '.join(missing)}. Place all CSVs in same folder as app.py.")
        st.stop()

@st.cache_data
def load_data():
    check_files()
    train = pd.read_csv(os.path.join(DATA_DIR, 'dataset.csv'))
    desc = pd.read_csv(os.path.join(DATA_DIR, 'symptom_Description.csv'))
    sev = pd.read_csv(os.path.join(DATA_DIR, 'Symptom-severity.csv'))
    prec = pd.read_csv(os.path.join(DATA_DIR, 'symptom_precaution.csv'))
    return train, desc, sev, prec

train_df, desc_df, sev_df, prec_df = load_data()

@st.cache_data
def train_model():
    symptom_columns = [col for col in train_df.columns if col.startswith('Symptom_')]
    all_symptoms = set()
    for col in symptom_columns:
        symptoms = train_df[col].dropna().apply(lambda x: x.strip().lower().replace(' ', '_'))
        all_symptoms.update(symptoms.unique())
    all_symptoms = sorted(list(all_symptoms))

    X = pd.DataFrame(0, index=train_df.index, columns=all_symptoms)
    for col in symptom_columns:
        for idx, symptom in train_df[col].items():
            if pd.notna(symptom):
                clean_symptom = symptom.strip().lower().replace(' ', '_')
                if clean_symptom in all_symptoms:
                    X.loc[idx, clean_symptom] = 1
    X = X.fillna(0)
    
    y = train_df['Disease']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, le, list(X.columns)

model, le, symptom_cols = train_model()

# Helper function to clean symptom names
def clean_symptom_name(symptom):
    return symptom.strip().lower().replace(' ', '_')

# ✅ Streamlit setup
st.title("🤖 AI Health Assistant")

if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Home", "Symptoms", "Results"], index=["Home", "Symptoms", "Results"].index(st.session_state.page))

# ✅ Home Page
if page == "Home":
    st.header("Welcome to AI Health Assistant")
    st.write("Get AI-based predictions for possible diseases based on your symptoms.")
    if st.button("🚀 Start Assessment"):
        st.session_state.page = "Symptoms"
        st.rerun()

# ✅ Symptoms Page
elif page == "Symptoms":
    st.header("🔍 Select Your Symptoms")
    search = st.text_input("Search symptoms...")
    display_symptoms = [s.replace('_', ' ').title() for s in symptom_cols]
    search_lower = search.lower()
    filtered_indices = [i for i, s in enumerate(display_symptoms) if search_lower in s.lower()] if search else range(len(display_symptoms))
    filtered_display = [display_symptoms[i] for i in filtered_indices]
    selected_display = st.multiselect("Choose symptoms", filtered_display)
    selected = [clean_symptom_name(s) for s in selected_display]

    if st.button("📊 Predict Diseases"):
        if not selected:
            st.warning("Please select at least one symptom.")
        else:
            # Build input vector
            input_vec = np.zeros(len(symptom_cols))
            for s in selected:
                if s in symptom_cols:
                    idx = symptom_cols.index(s)
                    input_vec[idx] = 1

            probs = model.predict_proba([input_vec])[0]
            sorted_idx = np.argsort(probs)[::-1]
            diseases = le.inverse_transform(sorted_idx)
            probs_sorted = probs[sorted_idx] * 100

            # Store in session_state
            st.session_state.diseases = diseases
            st.session_state.probs = probs_sorted
            st.session_state.selected = selected
            st.success("Prediction complete! Check the Results tab.")
            st.session_state.page = "Results"
            st.rerun()

# ✅ Results Page
elif page == "Results":
    st.header("📋 Your Health Assessment Results")
    st.info("These predictions are for educational purposes. Always consult a medical professional.")

    if 'diseases' in st.session_state:
        diseases = st.session_state.diseases
        probs = st.session_state.probs
        selected = st.session_state.selected
        threshold = 60  # Only show description/precautions for high-probability diseases

        # Highlight the top prediction prominently
        top_disease = diseases[0]
        top_prob = probs[0]
        # choose color/emoji based on top probability
        if top_prob < 30:
            top_risk, top_emoji = "Low", "🟢"
            top_bg = "#ecfdf5"
        elif top_prob < 60:
            top_risk, top_emoji = "Medium", "🟡"
            top_bg = "#fffbeb"
        else:
            top_risk, top_emoji = "High", "🔴"
            top_bg = "#fff1f2"

        st.markdown("### Top Prediction")
        # Simple styled card using markdown + inline styles
        disease_desc_top = desc_df[desc_df['Disease'] == top_disease]['Description'].iloc[0] if len(desc_df[desc_df['Disease'] == top_disease]) > 0 else ""
        disease_precautions_top = prec_df[prec_df['Disease'] == top_disease]

        st.markdown(f"<div style='background:{top_bg}; padding:16px; border-radius:8px; border:1px solid rgba(0,0,0,0.06)'>"
                    f"<h2 style='margin:0'>{top_disease} {top_emoji} <small style='font-size:14px;color:#374151'>({top_risk} Risk)</small></h2>"
                    f"<p style='margin:6px 0 4px 0; font-size:15px; color:#374151'><strong>Match Probability:</strong> {top_prob:.1f}%</p>"
                    f"</div>", unsafe_allow_html=True)

        st.progress(top_prob / 100)

        if disease_desc_top:
            st.markdown(f"**Description:** {disease_desc_top}")

        if len(disease_precautions_top) > 0:
            st.markdown("**Recommended Precautions:**")
            for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                if col in disease_precautions_top.columns and pd.notna(disease_precautions_top[col].iloc[0]):
                    st.write(f"- {disease_precautions_top[col].iloc[0]}")

        st.markdown("---")

        st.subheader("Other Possible Conditions")
        # iterate remaining diseases (skip top)
        for i, (dis, prob) in enumerate(zip(diseases[1:], probs[1:]), 2):
            # ✅ Compute risk & emoji per disease
            if prob < 30:
                risk, emoji = "Low", "🟢"
            elif prob < 60:
                risk, emoji = "Medium", "🟡"
            else:
                risk, emoji = "High", "🔴"

            st.markdown(f"### {i}. {dis} {emoji} ({risk} Risk)")
            st.progress(prob / 100)
            st.write(f"**Match Probability:** {prob:.1f}%")
            # Show description & precautions only for high-probability diseases (auto)
            if prob >= threshold:
                disease_desc = desc_df[desc_df['Disease'] == dis]['Description'].iloc[0] if len(desc_df[desc_df['Disease'] == dis]) > 0 else ""
                if disease_desc:
                    st.markdown(f"**Description:** {disease_desc}")

                disease_precautions = prec_df[prec_df['Disease'] == dis]
                if len(disease_precautions) > 0:
                    st.markdown("**Recommended Precautions:**")
                    for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                        if col in disease_precautions.columns and pd.notna(disease_precautions[col].iloc[0]):
                            st.write(f"- {disease_precautions[col].iloc[0]}")

            # Also provide a clickable expander so users can view details on demand
            with st.expander(f"View details for {dis}"):
                # Show description (if available)
                disease_desc2 = desc_df[desc_df['Disease'] == dis]['Description'].iloc[0] if len(desc_df[desc_df['Disease'] == dis]) > 0 else ""
                if disease_desc2:
                    st.markdown(f"**Description:** {disease_desc2}")
                else:
                    st.info("No description available for this disease.")

                # Show precautions (if available)
                disease_precautions2 = prec_df[prec_df['Disease'] == dis]
                if len(disease_precautions2) > 0:
                    st.markdown("**Recommended Precautions:**")
                    for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                        if col in disease_precautions2.columns and pd.notna(disease_precautions2[col].iloc[0]):
                            st.write(f"- {disease_precautions2[col].iloc[0]}")
                else:
                    st.info("No specific precautions found for this disease.")

            st.markdown("---")

    # ✅ Matched Symptoms
        st.markdown("### 🧠 Matched Symptoms")
        st.write(", ".join([s.replace('_', ' ').title() for s in selected]))

        # ✅ Add per-disease probability chart
        st.subheader("📊 Disease Probability Chart")
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['green' if p<30 else 'yellow' if p<60 else 'red' for p in probs]
        ax.barh(diseases, probs, color=colors)
        ax.set_xlabel("Match Probability (%)")
        ax.set_xlim(0, 100)
        for i, v in enumerate(probs):
            ax.text(v + 1, i, f"{v:.1f}%", va='center')
        st.pyplot(fig)

    else:
        st.info("Please complete a prediction first in the Symptoms tab.")

# Footer
st.markdown("---")
st.caption("⚕️ Powered by Streamlit & Scikit-learn | For educational use only.")
