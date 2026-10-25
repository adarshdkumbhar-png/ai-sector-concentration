import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Who Can Still Build Frontier AI?", layout="wide")

DATA_PATH = "notable_ai_models.csv"
SECTOR_ORDER = ["Industry", "Industry–academia collaboration", "Academia", "Other"]
SECTOR_COLOURS = ["#1f6feb", "#8250df", "#e3a008", "#9aa0a6"]


def classify(raw):
    labels = {x.strip() for x in str(raw).split(",") if x.strip() and x.strip() != "nan"}
    if labels == {"Industry"}:
        return "Industry"
    if labels == {"Academia"}:
        return "Academia"
    if {"Industry", "Academia"} <= labels:
        return "Industry–academia collaboration"
    return "Other"


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["Publication date"] = pd.to_datetime(df["Publication date"], errors="coerce")
    df = df.dropna(subset=["Publication date"])
    df["Year"] = df["Publication date"].dt.year
    df["Sector"] = df["Organization categorization"].apply(classify)
    df["Training compute (FLOP)"] = pd.to_numeric(df["Training compute (FLOP)"], errors="coerce")
    return df[(df["Year"] >= 2014) & (df["Year"] <= 2024)]


def sector_shares(df):
    counts = df.groupby(["Year", "Sector"]).size().unstack(fill_value=0).reindex(columns=SECTOR_ORDER, fill_value=0)
    totals = counts.sum(axis=1)
    shares = counts.div(totals, axis=0).mul(100)
    long = shares.reset_index().melt(id_vars="Year", var_name="Sector", value_name="Share (%)")
    long["Models"] = counts.reset_index().melt(id_vars="Year", value_name="n")["n"].values
    long["Total"] = long["Year"].map(totals)
    long["Rank"] = long["Sector"].map(SECTOR_ORDER.index)
    return long, counts, totals


df = load_data(DATA_PATH)
shares, counts, totals = sector_shares(df)

st.title("Who can still build frontier AI?")
st.markdown(
    "**Group question:** Is influential AI model development concentrated in fewer organisations?  \n"
    "**Persona:** Technology policy analyst  \n"
    "**This section asks:** has notable model development moved out of universities and into a single sector?"
)

ind_2014 = counts.loc[2014, "Industry"] / totals.loc[2014] * 100
ind_2024 = counts.loc[2024, "Industry"] / totals.loc[2024] * 100
aca_2024 = counts.loc[2024, "Academia"] / totals.loc[2024] * 100

c1, c2, c3 = st.columns(3)
c1.metric("Industry-only share of notable models, 2014", f"{ind_2014:.1f}%")
c2.metric("Industry-only share of notable models, 2024", f"{ind_2024:.1f}%", f"{ind_2024 - ind_2014:+.1f} pts")
c3.metric("Academia-only share of notable models, 2024", f"{aca_2024:.1f}%")

st.divider()
st.subheader("1. Who produces notable AI models each year?")
st.caption("Share of notable AI models by sector of the developing organisation(s), 2014–2024.")

start, end = st.slider("Year range", 2014, 2024, (2014, 2024))
view = shares[(shares["Year"] >= start) & (shares["Year"] <= end)]

bars = (
    alt.Chart(view)
    .mark_bar()
    .encode(
        x=alt.X("Year:O", title="Year"),
        y=alt.Y("Share (%):Q", stack="normalize", title="Share of notable models", axis=alt.Axis(format="%")),
        color=alt.Color("Sector:N", sort=SECTOR_ORDER, scale=alt.Scale(domain=SECTOR_ORDER, range=SECTOR_COLOURS), legend=alt.Legend(orient="bottom", title=None)),
        order=alt.Order("Rank:Q"),
        tooltip=["Year", "Sector", alt.Tooltip("Share (%):Q", format=".1f"), "Models", "Total"],
    )
    .properties(height=380)
)
st.altair_chart(bars, width="stretch")
st.markdown(
    f"**Reading:** in 2014 universities working alone produced most notable models. "
    f"By 2024 companies working alone produced {ind_2024:.0f}% of them, and academia on its own was down to {aca_2024:.0f}%."
)

st.divider()
st.subheader("2. Who can afford the compute?")
st.caption("Training compute of each notable model (log scale). Models without a published compute estimate are left out.")

threshold_label = st.selectbox("Frontier compute threshold", ["1e22 FLOP", "1e23 FLOP", "1e24 FLOP"], index=1)
threshold = float(threshold_label.split()[0])

pts = df.dropna(subset=["Training compute (FLOP)"])
pts = pts[pts["Training compute (FLOP)"] > 0]
above = pts[pts["Training compute (FLOP)"] >= threshold]
ind_above = (above["Sector"] == "Industry").mean() * 100 if len(above) else 0
aca_above = (above["Sector"] == "Academia").sum()

d1, d2 = st.columns(2)
d1.metric(f"Models at or above {threshold_label}", f"{len(above)}")
d2.metric("…built by industry alone", f"{ind_above:.1f}%")
st.caption(f"Built by academia alone: {aca_above}")

scatter = (
    alt.Chart(pts)
    .mark_circle(size=55, opacity=0.75)
    .encode(
        x=alt.X("Publication date:T", title="Publication date"),
        y=alt.Y("Training compute (FLOP):Q", scale=alt.Scale(type="log"), title="Training compute (FLOP, log scale)"),
        color=alt.Color("Sector:N", sort=SECTOR_ORDER, scale=alt.Scale(domain=SECTOR_ORDER, range=SECTOR_COLOURS), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Model", "Organization", "Sector", alt.Tooltip("Publication date:T", format="%b %Y"), alt.Tooltip("Training compute (FLOP):Q", format=".2e")],
    )
)
rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(strokeDash=[6, 4], color="#d1242f").encode(y="y:Q")
st.altair_chart((scatter + rule).properties(height=420), width="stretch")
st.markdown(
    "**Reading:** above the dashed line, almost every point is blue. The models that need the most compute "
    "are now built almost entirely by companies, which is a second kind of concentration: not just fewer organisations, but one sector."
)

st.divider()
with st.expander("Trace a number back to the source"):
    year = st.selectbox("Year", list(range(2024, 2013, -1)))
    sector = st.selectbox("Sector", SECTOR_ORDER)
    rows = df[(df["Year"] == year) & (df["Sector"] == sector)]
    st.write(f"{len(rows)} of {len(df[df['Year'] == year])} notable models in {year} → {len(rows) / max(len(df[df['Year'] == year]), 1) * 100:.1f}%")
    rows = rows.assign(**{"Publication date": rows["Publication date"].dt.date})
    st.dataframe(
        rows[["Model", "Organization", "Publication date", "Organization categorization", "Training compute (FLOP)"]].sort_values("Publication date"),
        width="stretch",
        hide_index=True,
    )

with st.expander("Data and method"):
    st.markdown(
        "- **Source:** Epoch AI, *Notable AI Models* dataset (`notable_ai_models.csv`), downloaded 25 October 2026 from epoch.ai/data/ai-models.\n"
        "- **Sector:** from the `Organization categorization` column. A model is *Industry* if every listed developer is industry, "
        "*Academia* if every developer is academic, *collaboration* if both appear, and *Other* for government, research collectives or blanks.\n"
        "- **Years:** 2014–2024, by `Publication date`.\n"
        "- **Compute:** `Training compute (FLOP)`; models without an estimate are excluded from section 2 only."
    )
