"""
Mental Health in Tech: interactive dashboard (Streamlit)
Data: OSMI Mental Health in Tech Survey 2014 (1,259 respondents).
Run:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import chi2_contingency

st.set_page_config(page_title="Mental health in tech", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------- constants
TEAL, RUST, GREY, AMBER = "#1d6f7a", "#c4553b", "#a9b4bb", "#d9a441"
ANSWER_COLORS = {"Yes": TEAL, "No": RUST, "Maybe": AMBER, "Not sure": AMBER,
                 "Don't know": GREY, "Some of them": AMBER}
SIZE_ORDER = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
SUPPORT_COLS = ["benefits", "care_options", "wellness_program", "seek_help", "anonymity"]
FRIENDLY = {  # readable names for the variable picker
    "family_history": "Family history of mental illness", "work_interfere": "Work interference",
    "benefits": "Employer mental health benefits", "care_options": "Knows care options",
    "wellness_program": "Wellness programme discussed", "seek_help": "Resources to seek help",
    "anonymity": "Anonymity protected", "leave": "Ease of taking leave", "Gender": "Gender",
    "age_group": "Age group", "no_employees": "Company size", "remote_work": "Works remotely",
    "tech_company": "Tech company", "self_employed": "Self-employed",
    "mental_health_consequence": "Fears mental health consequences",
    "phys_health_consequence": "Fears physical health consequences",
    "coworkers": "Would tell coworkers", "supervisor": "Would tell supervisor",
    "mental_health_interview": "Would mention mental health in interview",
    "phys_health_interview": "Would mention physical health in interview",
    "mental_vs_physical": "Employer treats mental = physical", "obs_consequence": "Saw negative consequences at work",
    "Country": "Country",
}

# ---------------------------------------------------------------- data
MALE = {"male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "man", "msle", "mail", "malr", "cis male",
        "cis man", "guy (-ish) ^_^", "ostensibly male, unsure what that really means", "something kinda male?"}
FEMALE = {"female", "f", "woman", "femake", "cis female", "female (cis)", "cis-female/femme", "femail",
          "female (trans)", "trans-female", "trans woman"}


def clean_gender(value):
    """Map ~50 free-text spellings to Male / Female / Other."""
    v = str(value).strip().lower()
    return "Male" if v in MALE else "Female" if v in FEMALE else "Other"


@st.cache_data(show_spinner="Loading data...")
def load_data(source):
    """Read and clean the survey. `source` is a file path or an uploaded file."""
    df = pd.read_csv(source)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Gender"] = df["Gender"].apply(clean_gender)
    bad = (df["Age"] < 18) | (df["Age"] > 75)                    # impossible ages -> median
    df.loc[bad, "Age"] = np.nan
    df["Age"] = df["Age"].fillna(df["Age"].median()).astype(int)
    df["self_employed"] = df["self_employed"].fillna("No")
    df["work_interfere"] = df["work_interfere"].fillna("Not applicable")   # only asked if a condition exists
    df["state"] = df["state"].fillna("Non-US")
    df["age_group"] = pd.cut(df["Age"], [17, 24, 34, 44, 54, 75], labels=["18-24", "25-34", "35-44", "45-54", "55+"])
    df["treated"] = (df["treatment"] == "Yes").astype(int)
    df["support_score"] = sum((df[c] == "Yes").astype(int) for c in SUPPORT_COLS)
    return df


def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2, p, _, _ = chi2_contingency(ct)
    return float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))), float(p)


def rate_table(data, col, order=None, min_n=1):
    g = data.groupby(col, observed=True)["treated"].agg(n="size", rate=lambda s: s.mean() * 100).reset_index()
    g = g[g["n"] >= min_n]
    if order:
        g[col] = pd.Categorical(g[col], [o for o in order if o in set(g[col])], ordered=True)
        g = g.sort_values(col)
    return g


def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      legend_title_text="", font=dict(size=13))
    return fig


def rate_chart(data, col, title, order=None, min_n=1, horizontal=False):
    """Bar chart of the % who sought treatment per category, with a baseline."""
    g = rate_table(data, col, order, min_n)
    g["label"] = g["rate"].round(0).astype(int).astype(str) + "%  (n=" + g["n"].astype(str) + ")"
    base = data["treated"].mean() * 100
    if horizontal:
        fig = px.bar(g, y=col, x="rate", text="label", orientation="h", title=title, color_discrete_sequence=[TEAL])
        fig.add_vline(x=base, line_dash="dash", line_color=GREY, annotation_text=f"All respondents {base:.0f}%")
        fig.update_xaxes(range=[0, 115], title="% who sought treatment")
    else:
        fig = px.bar(g, x=col, y="rate", text="label", title=title, color_discrete_sequence=[TEAL])
        fig.add_hline(y=base, line_dash="dash", line_color=GREY, annotation_text=f"All respondents {base:.0f}%")
        fig.update_yaxes(range=[0, 115], title="% who sought treatment")
    fig.update_traces(textposition="outside", cliponaxis=False)
    return style(fig)


def answer_mix(data, cols, labels, title):
    """100% stacked bars showing the answer mix for several related questions."""
    rows = []
    for c in cols:
        vc = data[c].value_counts(normalize=True) * 100
        rows += [{"Question": labels[c], "Answer": a, "Percent": v} for a, v in vc.items()]
    fig = px.bar(pd.DataFrame(rows), y="Question", x="Percent", color="Answer", orientation="h", title=title,
                 color_discrete_map=ANSWER_COLORS, text=None)
    fig.update_xaxes(title="% of respondents", range=[0, 100])
    return style(fig, 340)


# ---------------------------------------------------------------- load
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "survey.csv")
try:
    if os.path.exists(DATA_FILE):
        data = load_data(DATA_FILE)
    else:
        up = st.sidebar.file_uploader("Upload survey.csv", type="csv")
        if up is None:
            st.title("Mental health in tech")
            st.info("Upload survey.csv in the sidebar, or place it next to app.py.")
            st.stop()
        data = load_data(up)
except Exception as exc:  # unreadable / malformed file
    st.error(f"Could not load the survey data: {exc}")
    st.stop()

# ---------------------------------------------------------------- sidebar filters
st.sidebar.header("Filter respondents")
countries = data["Country"].value_counts().index.tolist()
sel_country = st.sidebar.multiselect("Country", countries, placeholder="All countries")
sel_gender = st.sidebar.multiselect("Gender", ["Male", "Female", "Other"], placeholder="All genders")
age_lo, age_hi = st.sidebar.slider("Age", 18, 75, (18, 75))
sel_size = st.sidebar.multiselect("Company size", SIZE_ORDER, placeholder="All sizes")
sel_tech = st.sidebar.radio("Employer type", ["All", "Tech company", "Non-tech company"])
sel_remote = st.sidebar.radio("Remote work", ["All", "Remote 50%+", "Mostly on-site"])

df = data.copy()
if sel_country: df = df[df["Country"].isin(sel_country)]
if sel_gender:  df = df[df["Gender"].isin(sel_gender)]
if sel_size:    df = df[df["no_employees"].isin(sel_size)]
df = df[df["Age"].between(age_lo, age_hi)]
if sel_tech != "All":   df = df[df["tech_company"] == ("Yes" if sel_tech == "Tech company" else "No")]
if sel_remote != "All": df = df[df["remote_work"] == ("Yes" if sel_remote == "Remote 50%+" else "No")]

# ---------------------------------------------------------------- header + KPIs
st.title("Mental health in tech")
st.caption("OSMI 2014 survey of 1,259 tech-industry respondents. Percentages describe survey participants, "
           "who were self-selected, so compare groups rather than reading them as industry-wide rates.")

if len(df) < 20:
    st.warning(f"Only {len(df)} respondents match these filters. Widen them to see reliable results.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Respondents", f"{len(df):,}", f"{len(df) / len(data) * 100:.0f}% of all")
k2.metric("Sought treatment", f"{df['treated'].mean() * 100:.1f}%")
k3.metric("Employer offers benefits", f"{(df['benefits'] == 'Yes').mean() * 100:.1f}%")
k4.metric("Fear mental health disclosure", f"{(df['mental_health_consequence'] == 'Yes').mean() * 100:.1f}%")

tabs = st.tabs(["Who responded", "What drives treatment", "Workplace support", "Stigma and disclosure", "Geography", "Data"])

# ---- Tab 1: profile
with tabs[0]:
    c1, c2 = st.columns(2)
    fig = px.histogram(df, x="Age", nbins=30, title="Age distribution", color_discrete_sequence=[TEAL])
    fig.add_vline(x=df["Age"].median(), line_dash="dash", annotation_text=f"Median {df['Age'].median():.0f}")
    c1.plotly_chart(style(fig), width="stretch")
    gc = df["Gender"].value_counts().reset_index()
    gc.columns = ["Gender", "Respondents"]
    fig = px.pie(gc, names="Gender", values="Respondents", hole=0.55, title="Gender",
                 color_discrete_sequence=[TEAL, RUST, GREY])
    c2.plotly_chart(style(fig), width="stretch")
    c3, c4 = st.columns(2)
    sc = df["no_employees"].value_counts().reindex(SIZE_ORDER).reset_index()
    sc.columns = ["Company size", "Respondents"]
    c3.plotly_chart(style(px.bar(sc, x="Company size", y="Respondents", title="Company size",
                                 color_discrete_sequence=[TEAL], text="Respondents")), width="stretch")
    wi = df["work_interfere"].value_counts().reindex(["Never", "Rarely", "Sometimes", "Often", "Not applicable"]).dropna().reset_index()
    wi.columns = ["Interference", "Respondents"]
    c4.plotly_chart(style(px.bar(wi, x="Interference", y="Respondents", title="Does a condition interfere with work?",
                                 color_discrete_sequence=[TEAL], text="Respondents")), width="stretch")

# ---- Tab 2: drivers
with tabs[1]:
    st.subheader("Treatment rate by any factor")
    var_keys = [k for k in FRIENDLY if k in df.columns]
    pick = st.selectbox("Choose a factor", var_keys, format_func=lambda k: FRIENDLY[k], index=0)
    orders = {"no_employees": SIZE_ORDER, "age_group": ["18-24", "25-34", "35-44", "45-54", "55+"],
              "work_interfere": ["Never", "Rarely", "Sometimes", "Often", "Not applicable"],
              "leave": ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"]}
    min_n = 5 if pick == "Country" else 1
    d = df[df["Country"].isin(df["Country"].value_counts().head(10).index)] if pick == "Country" else df
    st.plotly_chart(rate_chart(d, pick, f"% who sought treatment by: {FRIENDLY[pick].lower()}",
                               orders.get(pick), min_n, horizontal=(pick == "Country")), width="stretch")
    st.caption("The dashed line is the treatment rate across the currently filtered respondents. Small groups (low n) are unreliable.")

    st.subheader("Which factors matter most?")
    rows = []
    for k in var_keys:
        if df[k].nunique() > 1 and df["treatment"].nunique() > 1:
            v, p = cramers_v(df[k], df["treatment"])
            rows.append({"Factor": FRIENDLY[k], "Association (Cramér's V)": round(v, 3), "Significant": "p < 0.05" if p < 0.05 else "Not significant"})
    rank = pd.DataFrame(rows).sort_values("Association (Cramér's V)")
    fig = px.bar(rank, y="Factor", x="Association (Cramér's V)", color="Significant", orientation="h",
                 color_discrete_map={"p < 0.05": TEAL, "Not significant": GREY}, title="Strength of association with seeking treatment")
    st.plotly_chart(style(fig, 640), width="stretch")
    st.info("Work interference is only asked of people who have a condition, so its top ranking is partly built into the survey. "
            "Family history and awareness of care options are the strongest actionable signals. Association is not causation.")

# ---- Tab 3: support
with tabs[2]:
    labels = {"benefits": "Mental health benefits", "care_options": "Knows care options", "wellness_program": "Wellness programme discussed",
              "seek_help": "Resources to seek help", "anonymity": "Anonymity protected"}
    st.plotly_chart(answer_mix(df, SUPPORT_COLS, labels, "How well do employers support and communicate?"), width="stretch")
    c1, c2 = st.columns(2)
    ct = (pd.crosstab(df["no_employees"], df["benefits"], normalize="index") * 100).reindex(SIZE_ORDER).dropna(how="all")
    ct = ct.reset_index().melt(id_vars="no_employees", var_name="Answer", value_name="Percent")
    fig = px.bar(ct, x="no_employees", y="Percent", color="Answer", title="Benefits by company size",
                 color_discrete_map=ANSWER_COLORS, category_orders={"no_employees": SIZE_ORDER})
    c1.plotly_chart(style(fig.update_xaxes(title="Company size")), width="stretch")
    c2.plotly_chart(rate_chart(df, "support_score", "Treatment rate by number of confirmed support features (0-5)"), width="stretch")
    st.caption("Employees who can confirm more support features are more likely to have sought treatment, and the big 'Don't know' "
               "segments show that communicating existing benefits is a low-cost win.")

# ---- Tab 4: stigma
with tabs[3]:
    c1, c2 = st.columns(2)
    cons = pd.DataFrame({"Mental health issue": df["mental_health_consequence"].value_counts(normalize=True) * 100,
                         "Physical health issue": df["phys_health_consequence"].value_counts(normalize=True) * 100})
    cons = cons.reindex(["Yes", "Maybe", "No"]).reset_index().melt(id_vars="index", var_name="Topic", value_name="Percent")
    fig = px.bar(cons, x="index", y="Percent", color="Topic", barmode="group", title="Would disclosing have negative consequences?",
                 color_discrete_sequence=[RUST, TEAL], text=cons["Percent"].round(0).astype(int).astype(str) + "%")
    c1.plotly_chart(style(fig.update_xaxes(title="")), width="stretch")
    disc = pd.DataFrame({"Scenario": ["Supervisor (mental)", "Coworkers (mental)", "Interview: physical", "Interview: mental"],
                         "Percent": [(df["supervisor"] == "Yes").mean() * 100, (df["coworkers"] == "Yes").mean() * 100,
                                     (df["phys_health_interview"] == "Yes").mean() * 100, (df["mental_health_interview"] == "Yes").mean() * 100]}
                        ).sort_values("Percent")
    fig = px.bar(disc, y="Scenario", x="Percent", orientation="h", title="Who would employees tell? (% Yes)",
                 color_discrete_sequence=[TEAL], text=disc["Percent"].round(1).astype(str) + "%")
    c2.plotly_chart(style(fig.update_xaxes(range=[0, 60])), width="stretch")
    mvp = (pd.crosstab(df["tech_company"].map({"Yes": "Tech company", "No": "Non-tech company"}), df["mental_vs_physical"], normalize="index") * 100)
    mvp = mvp.reset_index().melt(id_vars="tech_company", var_name="Answer", value_name="Percent")
    fig = px.bar(mvp, x="tech_company", y="Percent", color="Answer", barmode="group", color_discrete_map=ANSWER_COLORS,
                 title="Does your employer take mental health as seriously as physical health?")
    st.plotly_chart(style(fig.update_xaxes(title="")), width="stretch")

# ---- Tab 5: geography
with tabs[4]:
    g = rate_table(df, "Country", min_n=10)
    if g.empty:
        st.info("No country has 10 or more respondents under the current filters.")
    else:
        fig = px.choropleth(g, locations="Country", locationmode="country names", color="rate", hover_data={"n": True, "rate": ":.0f"},
                            color_continuous_scale="Teal", range_color=(0, 100), title="Treatment rate by country (countries with 10+ respondents)")
        fig.update_layout(coloraxis_colorbar_title="% treated")
        st.plotly_chart(style(fig, 460), width="stretch")
    us = df[df["state"] != "Non-US"]
    gs = rate_table(us, "state", min_n=15)
    if len(gs):
        fig = px.choropleth(gs, locations="state", locationmode="USA-states", scope="usa", color="rate", hover_data={"n": True, "rate": ":.0f"},
                            color_continuous_scale="Teal", range_color=(0, 100), title="Treatment rate by US state (states with 15+ respondents)")
        st.plotly_chart(style(fig, 460), width="stretch")
    st.caption("Country and state samples are small outside the USA, UK and Canada. Treat differences as directional.")

# ---- Tab 6: data
with tabs[5]:
    st.write(f"{len(df):,} rows match the current filters.")
    shown = df.drop(columns=["comments", "treated"], errors="ignore")
    st.dataframe(shown, width="stretch", height=420)
    st.download_button("Download filtered data (CSV)", shown.to_csv(index=False).encode("utf-8"), "mental_health_filtered.csv", "text/csv")
