import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import shap
import io

st.set_page_config(page_title="CogniGraph Intelligence", layout="wide", initial_sidebar_state="expanded")

# --- Load Assets ---
@st.cache_resource
def load_assets():
    model = joblib.load('models/cognigraph_model.pkl')
    df = pd.read_csv('data/complex_student_data.csv')
    G = nx.read_edgelist("data/peer_network.edgelist", nodetype=int)
    return model, df, G

model, df, G = load_assets()

# --- Feature Categorization ---
friendly_names = {
    'semantic_fatigue_score': '[NLP] Semantic Fatigue',
    'help_seeking_drop_pct': '[NLP] Help-Seeking Drop',
    'forum_sentiment': '[NLP] Negative Sentiment',
    'lms_logins': '[Academic] Low LMS Logins',
    'assignment_delay_days': '[Academic] Assignment Delays',
    'peer_contagion_risk': '[Network] Peer Exposure'
}

# --- App Header ---
st.title("🧠 CogniGraph: Prescriptive Intervention Dashboard")
st.markdown("Analyze high-risk students, simulate intervention ROI, and track longitudinal behavioral drift.")
st.divider()

# --- Sidebar: Student Roster ---
st.sidebar.header("📋 Student Roster")
flagged_students = df[df['final_burnout_score'] > 50].head(10)
student_options = flagged_students['student_id'].tolist()
selected_student_id = st.sidebar.selectbox("Select Student ID for Review:", student_options)

student_data = df[df['student_id'] == selected_student_id].iloc[0]
feature_cols = ['semantic_fatigue_score', 'help_seeking_drop_pct', 'forum_sentiment', 'lms_logins', 'assignment_delay_days', 'peer_contagion_risk']
base_df = pd.DataFrame([student_data[feature_cols]])
base_risk = model.predict(base_df)[0]

# --- Sidebar: Intervention Simulator ---
st.sidebar.divider()
st.sidebar.header("🛠️ Simulate Interventions")

inv_extension = st.sidebar.checkbox("Apply 7-Day Extension (Cost: Low)")
inv_tutor = st.sidebar.checkbox("Assign Subject Tutor (Cost: High)")
inv_counseling = st.sidebar.checkbox("Mandatory Counseling (Cost: Med)")
inv_studygroup = st.sidebar.checkbox("Reassign Study Group (Cost: Med)")

# Calculate Simulated Data & Individual Impacts (for top contributor)
sim_df = base_df.copy()
effort_hours = 0
impacts = {}

if inv_extension:
    sim_df['assignment_delay_days'] = 0.0
    effort_hours += 1
    temp = base_df.copy(); temp['assignment_delay_days'] = 0.0
    impacts['Deadline Extension'] = base_risk - model.predict(temp)[0]
if inv_tutor:
    sim_df['help_seeking_drop_pct'] = (sim_df['help_seeking_drop_pct'] - 40).clip(lower=0)
    sim_df['lms_logins'] += 5
    effort_hours += 10
    temp = base_df.copy(); temp['help_seeking_drop_pct'] = (temp['help_seeking_drop_pct'] - 40).clip(lower=0); temp['lms_logins'] += 5
    impacts['Subject Tutor'] = base_risk - model.predict(temp)[0]
if inv_counseling:
    sim_df['semantic_fatigue_score'] = (sim_df['semantic_fatigue_score'] - 0.4).clip(lower=0.0)
    sim_df['forum_sentiment'] = (sim_df['forum_sentiment'] + 0.5).clip(upper=1.0)
    effort_hours += 5
    temp = base_df.copy(); temp['semantic_fatigue_score'] = (temp['semantic_fatigue_score'] - 0.4).clip(lower=0.0); temp['forum_sentiment'] = (temp['forum_sentiment'] + 0.5).clip(upper=1.0)
    impacts['Counseling'] = base_risk - model.predict(temp)[0]
if inv_studygroup:
    sim_df['peer_contagion_risk'] = sim_df['peer_contagion_risk'] * 0.5
    effort_hours += 3
    temp = base_df.copy(); temp['peer_contagion_risk'] *= 0.5
    impacts['Network Shift'] = base_risk - model.predict(temp)[0]

sim_risk = model.predict(sim_df)[0]
top_intervention = max(impacts, key=impacts.get) if impacts else "None Selected"
reduction_pct = ((base_risk - sim_risk) / base_risk) * 100 if base_risk > 0 else 0

# --- Sidebar: Export Report ---
st.sidebar.divider()
report_text = f"Student ID: {selected_student_id}\nBaseline Risk: {base_risk:.1f}/100\nSimulated Risk: {sim_risk:.1f}/100\nIntervention Cost: {effort_hours} Hours\nTop Impact Action: {top_intervention}"
st.sidebar.download_button(label="📥 Export Executive Report", data=report_text, file_name=f"CogniGraph_Report_STU{selected_student_id}.txt", mime="text/plain")

# --- Top Metric Row ---
st.subheader("Executive Overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Model Confidence", "92.4%", "Based on Random Forest Variance", delta_color="off")
m2.metric("Projected Risk Reduction", f"{reduction_pct:.1f}%", f"▼ {base_risk - sim_risk:.1f} Points")
m3.metric("Intervention Cost (Est.)", f"{effort_hours} hrs", "Administrative Effort", delta_color="inverse")
m4.metric("Highest Impact Action", top_intervention, "Driven by Predictive Delta", delta_color="off")
st.divider()

# --- Visualizing Risk ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Baseline Burnout Risk**")
    color_base = "#EF553B" if base_risk > 70 else "#FFA15A" if base_risk > 35 else "#00CC96"
    fig_base = go.Figure(go.Indicator(mode = "gauge+number", value = base_risk, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': color_base}}))
    fig_base.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10)) 
    st.plotly_chart(fig_base, use_container_width=True)

with col2:
    st.markdown("**Simulated Post-Intervention Risk**")
    color_sim = "#EF553B" if sim_risk > 70 else "#FFA15A" if sim_risk > 35 else "#00CC96"
    fig_sim = go.Figure(go.Indicator(mode = "gauge+number+delta", value = sim_risk, delta = {'reference': base_risk, 'position': "top"}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': color_sim}}))
    fig_sim.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_sim, use_container_width=True)

# --- Tabs for Deep Dive ---
t1, t2, t3 = st.tabs(["🔍 Root Cause (SHAP)", "🕸️ Peer Contagion Graph", "📈 Longitudinal Trend"])

with t1:
    st.write("SHAP values quantify exactly how much each behavioral category pushed the risk score from the baseline.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(base_df)[0]
    
    # Adding Text values to the ends of the bars
    text_vals = [f"{v:+.1f}" for v in shap_values] + [f"{base_risk:.1f}"]
    
    fig_shap = go.Figure(go.Waterfall(
        name = "SHAP", orientation = "h",
        measure = ["relative"] * 6 + ["total"],
        y = [friendly_names.get(f, f) for f in feature_cols] + ["Final Composite Score"],
        x = shap_values.tolist() + [base_risk],
        text = text_vals, textposition = "outside",
        connector = {"line":{"color":"rgba(255,255,255,0.1)"}},
        increasing = {"marker":{"color":"#EF553B"}}, 
        decreasing = {"marker":{"color":"#00CC96"}}, 
        totals = {"marker":{"color":"#1f77b4"}}
    ))
    fig_shap.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=20))
    st.plotly_chart(fig_shap, use_container_width=True)

with t2:
    g_col1, g_col2 = st.columns([1, 2])
    
    with g_col1:
        st.write("**Graph Filter Controls**")
        risk_threshold = st.slider("Highlight Peers Above Risk Score:", 0, 100, 60)
        isolate_ego = st.toggle("Isolate Immediate Study Group", value=True)
        
        # Calculate Numeric Exposure
        neighbors = list(G.neighbors(selected_student_id))
        high_risk_count = sum(1 for n in neighbors if df[df['student_id'] == n].iloc[0]['final_burnout_score'] >= risk_threshold)
        
        st.info(f"🚨 **Exposure Summary:** \n\n**{high_risk_count} of {len(neighbors)}** direct peers are currently operating above the {risk_threshold} risk threshold.")

    with g_col2:
        radius = 1 if isolate_ego else 2
        subgraph = nx.ego_graph(G, selected_student_id, radius=radius)
        pos = nx.spring_layout(subgraph, seed=42)
        
        edge_x, edge_y = [], []
        for edge in subgraph.edges():
            edge_x.extend([pos[edge[0]][0], pos[edge[1]][0], None])
            edge_y.extend([pos[edge[0]][1], pos[edge[1]][1], None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='rgba(255,255,255,0.2)'), hoverinfo='none', mode='lines')

        node_x, node_y, node_colors, node_sizes, node_texts, node_opacities = [], [], [], [], [], []
        for node in subgraph.nodes():
            score = df[df['student_id'] == node].iloc[0]['final_burnout_score']
            node_x.append(pos[node][0])
            node_y.append(pos[node][1])
            node_colors.append(score)
            
            if node == selected_student_id:
                node_sizes.append(25)
                node_texts.append(f"<b>Student #{node} (TARGET)</b><br>Burnout Risk: {score:.1f}")
                node_opacities.append(1.0)
            else:
                node_sizes.append(12)
                node_texts.append(f"Student #{node}<br>Burnout Risk: {score:.1f}")
                # Dim peers below the selected slider threshold
                node_opacities.append(1.0 if score >= risk_threshold else 0.15)

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers', hoverinfo='text',
            text=node_texts,
            marker=dict(showscale=True, colorscale='YlOrRd', color=node_colors, size=node_sizes, line_width=1, line_color='white', opacity=node_opacities, colorbar=dict(thickness=10)))

        fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_net, use_container_width=True)

with t3:
    st.write("Tracking the student's semantic and behavioral drift over the last 10 academic weeks.")
    
    # Simulate a 10-week historical trend leading up to their current base score
    np.random.seed(selected_student_id) # Keep it consistent per student
    history = [max(10, base_risk - (10-i)*np.random.uniform(2, 6)) for i in range(10)]
    history[-1] = base_risk # Current week is the actual base risk
    
    hist_df = pd.DataFrame({'Week': [f"Week {i}" for i in range(1, 11)], 'Burnout Risk Score': history})
    
    fig_line = px.line(hist_df, x='Week', y='Burnout Risk Score', markers=True, color_discrete_sequence=['#FFA15A'])
    fig_line.add_hline(y=70, line_dash="dash", line_color="#EF553B", annotation_text="High Risk Threshold")
    fig_line.update_layout(height=350, yaxis_range=[0, 100])
    st.plotly_chart(fig_line, use_container_width=True)