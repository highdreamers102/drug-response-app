import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


st.set_page_config(page_title="Drug Response Explorer", layout = "wide")
@st.cache_data
def Load_data():
    return pd.read_csv("data/real_drug_dataset.csv")
df = Load_data()

page = st.sidebar.radio(
    "Page",
    ["Explorer","Side-effects Predictor ","Improvement Estimator","Methodology"]
)

if page == "Explorer":
    st.title("Drug Response Explorer")
    st.caption("Synthetic Dataset For Learning Only. Not For Medical Use")

    cond = st.sidebar.multiselect("Condition", df["Condition"].unique(), default = list(df["Condition"].unique())
    )
    filtered = df[df["Condition"].isin(cond)]
    st.write(f"{len(filtered)} Patients ")
    fig = px.box(filtered,x = "Drug_Name", y = "Improvement_Score")
    st.plotly_chart(fig, use_container_width = True)

elif page == "Side-effects Predictor ":
    st.title("Side-effects Predictor")
    drug = st.selectbox("Drug", sorted(df["Drug_Name"].unique()))
    sub = df[df["Drug_Name"] == drug]
    out = sub["Side_Effects"].value_counts(normalize=True).rename("Probability")
    st.caption(f"Based on {len(sub)} patients taking {drug}. Synthetic data.")
    st.bar_chart(out)


elif page == "Improvement Estimator":
    st.title("Improvement Estimator")
    st.warning("On this data not a single model could perform better or crossed average (R2 ≈ 0),this is not prediction only average")
    drug = st.selectbox("Drug", sorted(df["Drug_Name"].unique()))
    sub = df[df["Drug_Name"] == drug]["Improvement_Score"]
    margin = 1.96*sub.std()/len(sub) ** 0.5

    c1,c2 = st.columns(2)
    c1.metric(f"{drug} average of drug",f"{sub.mean():.2f}")
    c2.metric(f"average of all drugs", f"{df['Improvement_Score'].mean():.2f}")
    st.caption(f"95% range of average {sub.mean()-margin:.2f} to {sub.mean()+margin:.2f}({len(sub)} patients)")

else:
    st.title("Methodology")
    st.subheader("1. Does anything predict Improvement_Score?")
    st.write("One-way ANOVA on mean Improvement_Score across groups:")
    st.table(pd.DataFrame({
        "Grouping": ["Drug_Name", "Condition", "Gender"],
        "p-value": [0.947, 0.719, 0.771],
        "Result": ["No significant difference"] * 3
    }))

    st.subheader("2. Model comparison (5-fold cross-validation)")
    st.table(pd.DataFrame({"Model": ["Baseline (always predict mean)", "Linear Regression", "Random Forest"],
            "R²": [-0.012, -0.039, -0.113],
            "MAE": [1.155, 1.168, 1.201]
    }))

    st.write("No model beat the baseline. A control run with the target shuffled scored "
             "the same as the real data, so the target carries no learnable signal.")

    st.subheader("3. Data limitations")
    st.markdown("""
- The dataset appears synthetic: every drug uses the same five doses (50-850 mg).
- Several doses are clinically unrealistic, and insulin is dosed in units, not mg.
- Age and treatment duration are uniformly distributed, unlike real patients.
- `Side_Effects` is only known after treatment, so it was excluded from the improvement analysis to avoid data leakage.
- For learning only. Not medical advice.
""")

