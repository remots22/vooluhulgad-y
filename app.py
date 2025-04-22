import streamlit as st
from utility import check_password
import pandas as pd
import numpy as np
import altair as alt
import os
from pathlib import Path

#parooli kontrollimine - meetod utility.py
if not check_password():
    st.stop()

def load_data():
    csv_path = Path(__file__).parent / "vooluhulk1923_2024.csv"
    df = pd.read_csv(
        csv_path, 
        sep=';', 
        decimal=',', 
        usecols=[0, 1], 
        header=0, 
        names=["date", "flow"],
        encoding='latin1'
    )
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['year'] = df['date'].dt.year
    return df

@st.cache_data
def get_data():
    return load_data()

df = get_data()
st.sidebar.header("Aastad")
years = sorted(df['year'].dropna().unique())
sel_mode = st.sidebar.radio("Vali üksik aasta või vahemik", ["Üksik aasta", "Vahemik"])
if sel_mode == "Üksik aasta":
    year = st.sidebar.selectbox("Vali aasta", years)
    start_year = end_year = year
else:
    start_year, end_year = st.sidebar.select_slider("Vali vahemik", options=years, value=(years[0], years[-1]))
st.sidebar.header("Veehaarde parameetrid")
HEJ = st.sidebar.slider("HEJ", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
VTT = st.sidebar.slider("VTT", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
KP = st.sidebar.slider("KP", min_value=0.0, max_value=1.0, value=0.6, step=0.01)
threshold_val = HEJ + VTT + KP
threshold_fmt = f"{threshold_val:.2f}".replace('.', ',')
st.markdown(f"HEJ + BTT + KALAPÄÄS = {threshold_fmt}", unsafe_allow_html=True)
if start_year > end_year:
    start_year, end_year = end_year, start_year
selected = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
grouped = selected.groupby('year')['flow'].apply(list)
leap_years = [y for y, vals in grouped.items() if len(vals) == 366]
if leap_years:
    st.markdown(
        f"<span style='color:gray'>Vahemikku jäävad järgmised liigaastad: {', '.join(map(str, leap_years))}</span>",
        unsafe_allow_html=True
    )
sorted_lists = []
for y, values in grouped.items():
    sorted_vals = sorted(values, reverse=True)
    sorted_lists.append(sorted_vals[:365])
avg_flows = np.mean(sorted_lists, axis=0)
days = np.arange(1, len(avg_flows) + 1)
y_max_main = max(max(avg_flows), threshold_val) * 1.1
x_max_main = int(days[-1])
chart_df = pd.DataFrame({'Päev': days, 'Keskmine vooluhulk': avg_flows})
chart = alt.Chart(chart_df).mark_line().encode(
    x=alt.X('Päev:Q', title='Päev', scale=alt.Scale(domain=[0, x_max_main])),
    y=alt.Y('Keskmine vooluhulk:Q', title='Keskmine vooluhulk (m³/s)', scale=alt.Scale(domain=[0, y_max_main])),
    tooltip=[alt.Tooltip('Päev:Q', title='Päev'), alt.Tooltip('Keskmine vooluhulk:Q', title='Keskmine vooluhulk (m³/s)')]
).properties(title=f'Sorteeritud keskmised vooluhulgad (laskuv) ({start_year}-{end_year})', height=400)
threshold = HEJ + VTT + KP
intersect_idx = np.where(avg_flows <= threshold)[0]
if len(intersect_idx) > 0:
    intercept_day = int(days[intersect_idx[0]])
    dot = alt.Chart(pd.DataFrame({'Päev': [intercept_day], 'Keskmine vooluhulk': [threshold]})).mark_point(color='red', size=60).encode(x='Päev:Q', y='Keskmine vooluhulk:Q')
    chart = chart + dot
    days_unaffected = intercept_day - 1
    percent_unaffected = int(round(days_unaffected / 365 * 100))
    max_flow = max(avg_flows)
    area_df = pd.DataFrame({'x1': [intercept_day], 'x2': [len(days)], 'y1': [0], 'y2': [VTT]})
    area = alt.Chart(area_df).mark_area(color='yellow', opacity=0.3).encode(x='x1:Q', x2='x2:Q', y='y1:Q', y2='y2:Q')
    chart = chart + area
    vline = alt.Chart(pd.DataFrame({'Päev': [intercept_day]})).mark_rule(color='red').encode(x='Päev:Q')
    chart = chart + vline
    ann_text = f"{days_unaffected} PÄEVA ({percent_unaffected}%) AASTAS EI MÕJUTA BTT\nSILLAORU LÄVENDIS ENERGEETILIST\n\nPOTENSIAALI"
    ann = alt.Chart(pd.DataFrame({'Päev': [intercept_day], 'Keskmine vooluhulk': [max_flow], 'text': [ann_text]})).mark_text(align='left', dx=5, dy=-10, color='red').encode(x='Päev:Q', y='Keskmine vooluhulk:Q', text='text:N')
    chart = chart + ann
    days_left = len(days) - intercept_day
else:
    days_left = len(days)
threshold_line = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_rule(color='teal').encode(y='threshold:Q')
threshold_text = alt.Chart(pd.DataFrame({'threshold': [threshold]})).mark_text(color='teal', align='left', dx=5, dy=5).encode(
    x=alt.value(10),
    y='threshold:Q',
    text=alt.value(f"HEJ + BTT + KALAPÄÄS = {threshold_fmt}")
)
chart = chart + threshold_line + threshold_text
selector = alt.selection_point(fields=['Päev'], nearest=True, on='mouseover', empty='none')
selectors = alt.Chart(chart_df).mark_point().encode(
    x='Päev:Q',
    opacity=alt.value(0)
).add_params(selector)
hover_points = alt.Chart(chart_df).mark_point(color='orange', size=100).encode(
    x='Päev:Q',
    y='Keskmine vooluhulk:Q',
    tooltip=[alt.Tooltip('Päev:Q', title='Päev'), alt.Tooltip('Keskmine vooluhulk:Q', title='Keskmine vooluhulk (m³/s)')]
).transform_filter(selector)
hover_rules = alt.Chart(chart_df).mark_rule(color='gray').encode(
    x='Päev:Q'
).transform_filter(selector)
chart = chart + selectors + hover_points + hover_rules

tabs = st.tabs(["Graafik", "Vaata andmeid"])

with tabs[0]:
    st.altair_chart(chart, use_container_width=True)
    if len(intersect_idx) > 0:
        st.markdown(
            f"<span style='color:gray'>BTT ei mõjuta {days_unaffected} päeva ({percent_unaffected}%) aastas.</span>",
            unsafe_allow_html=True
        )
        g = 9.81
        H_const = 8.0
        efficiency = 0.8
        P_kw = g * H_const * VTT * efficiency
        P_str = f"P = {g} x {H_const} x {VTT} x {efficiency} = {round(P_kw,2):.2f} kW".replace(".", ",")
        st.markdown(f"*{P_str}*", unsafe_allow_html=True)
        P_kw_2 = round(P_kw, 2)
        P_kw_str2 = f"{P_kw_2:.2f}".replace(".", ",")
        potential_kwh = P_kw_2 * 24 * days_left
        potential_display = f"{potential_kwh:,.0f}".replace(",", " ")
        equation_str = f"Mõjupäevade potentsiaal = {P_kw_str2} x 24 x {days_left} = {potential_display} kWh/a"
        st.markdown(f"<strong><u>{equation_str}</u></strong>", unsafe_allow_html=True)
    if len(grouped) == 1:
        only_year = list(grouped.keys())[0]
        year_potentials = {only_year: potential_kwh}
    else:
        year_potentials = {}
        for y, vals in grouped.items():
            sorted_vals = sorted(vals, reverse=True)[:365]
            intercept_idx = next((i for i, v in enumerate(sorted_vals) if v <= threshold), len(sorted_vals))
            days_left_y = len(sorted_vals) - intercept_idx
            year_potentials[y] = P_kw_2 * 24 * days_left_y
    max_year = max(year_potentials, key=year_potentials.get)
    min_year = min(year_potentials, key=year_potentials.get)
    max_fmt = f"{year_potentials[max_year]:,.0f}".replace(",", " ")
    min_fmt = f"{year_potentials[min_year]:,.0f}".replace(",", " ")
    st.markdown(f"{max_year} aasta kõrgeim potentsiaal: {max_fmt} kWh/a", unsafe_allow_html=True)
    st.markdown(f"{min_year} aasta madalaim potentsiaal: {min_fmt} kWh/a", unsafe_allow_html=True)
    pot_df = pd.DataFrame({
        'Aasta': list(year_potentials.keys()),
        'Potentsiaal': list(year_potentials.values())
    })
    min_pot = pot_df['Potentsiaal'].min() # Minimaalne aastane potentsiaal
    max_pot = pot_df['Potentsiaal'].max() # Maksimaalne aastane potentsiaal
    years_dom = sorted(year_potentials.keys()) # Aastate sorteerimine x jaoks
    min_dom, max_dom = years_dom[0], years_dom[-1] # X piirid
    pot_scatter = alt.Chart(pot_df).mark_point(color='teal', size=60).encode(
        x=alt.X('Aasta:Q', title='Aasta', scale=alt.Scale(domain=[min_dom, max_dom])),
        y=alt.Y('Potentsiaal:Q', title='Mõjupäevade potentsiaal (kWh/a)', scale=alt.Scale(domain=[min_pot, max_pot]))
    ) # Scatter jaoks iga aasta potentsiaal
    pot_line = alt.Chart(pot_df).transform_regression(
        'Aasta', 'Potentsiaal', method='linear'
    ).mark_line(color='red', strokeDash=[4,4]).encode(
        x=alt.X('Aasta:Q', scale=alt.Scale(domain=[min_dom, max_dom])),
        y=alt.Y('Potentsiaal:Q', scale=alt.Scale(domain=[min_pot, max_pot]))
    ) # Trendijoon / lineaarne regressioon
    pot_chart = pot_scatter + pot_line # Punktide ja trendi kombineerimine
    # Graafiku kuvamine
    st.altair_chart(pot_chart, use_container_width=True)

with tabs[1]:
    st.markdown(f"### Toored andmed {start_year} - {end_year}")
    raw_filtered = df[(df['year'] >= start_year) & (df['year'] <= end_year)][['date', 'flow']]
    raw_filtered["date"] = raw_filtered["date"].dt.strftime("%Y-%m-%d")
    raw_filtered = raw_filtered.rename(columns={"date": "Kuupäev", "flow": "Vooluhulk (m³/s)"})
    st.dataframe(raw_filtered, use_container_width=True)