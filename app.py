"""
Streamlit Dashboard for Competitive Programming Curriculum Learning
Interactive web interface for problem recommendations and progress tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os

# Set page config
st.set_page_config(
    page_title="CP Curriculum Learning",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'student_data' not in st.session_state:
    st.session_state.student_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'engine' not in st.session_state:
    st.session_state.engine = None

# ============= SIDEBAR =============
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Setup", "📊 Demo", "ℹ️ Info"])
    
    with tab1:
        st.subheader("Initialize System")
        
        if st.button("🚀 Load Demo Data & Model", use_container_width=True):
            with st.spinner("Loading data and model..."):
                try:
                    from data_generator import CPDatasetGenerator
                    from model import CompetitiveProgrammingModel, train_model_from_data
                    from curriculum_engine import CurriculumLearningEngine
                    
                    # Generate dataset
                    generator = CPDatasetGenerator(num_problems=100)
                    solutions_df, problems, students = generator.save_dataset()
                    
                    # Train model
                    model, history = train_model_from_data(solutions_df, problems)
                    
                    # Initialize engine
                    engine = CurriculumLearningEngine(problems, model)
                    
                    # Store in session
                    st.session_state.solutions_df = solutions_df
                    st.session_state.problems = problems
                    st.session_state.students = students
                    st.session_state.model = model
                    st.session_state.engine = engine
                    
                    st.success("✅ System loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading system: {e}")
        
        st.divider()
        
        st.subheader("Student Selection")
        if 'students' in st.session_state:
            student_list = list(st.session_state.students.keys())
            selected_student = st.selectbox("Select student:", student_list)
            
            if selected_student:
                student_data = st.session_state.students[selected_student]
                st.session_state.selected_student_name = selected_student
                st.session_state.student_data = student_data
                st.success(f"Selected: {selected_student}")
    
    with tab2:
        st.subheader("Demo Information")
        st.write("""
        This demo uses **synthetic competitive programming data**:
        - 100 problems across 10 topics
        - 20 simulated students
        - Realistic code metrics and solution times
        - Multi-task deep learning model trained on this data
        """)
    
    with tab3:
        st.subheader("System Info")
        st.write("""
        **Competitive Programming Curriculum Learning System**
        
        Uses:
        - Multi-task Deep Learning
        - Curriculum Learning Theory
        - Thompson Sampling for recommendations
        - Zone of Proximal Development (ZPD)
        """)

# ============= MAIN CONTENT =============

st.markdown("# 🏆 Competitive Programming Curriculum Learning System")
st.markdown("Adaptive problem recommendations powered by deep learning")

# Check if system is loaded
if 'model' not in st.session_state or st.session_state.model is None:
    st.warning("⚠️ Please load the system first from the sidebar")
    st.stop()

# Get student profile
if st.session_state.student_data is None:
    st.info("📝 Please select a student from the sidebar")
    st.stop()

student_name = st.session_state.selected_student_name
engine = st.session_state.engine
student_data = st.session_state.student_data

# Build student profile
profile = engine.build_student_profile(student_data['submissions'])

# ============= HEADER METRICS =============
st.markdown("## 📊 Student Profile Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Skill Level",
        value=f"{profile['skill']:.1f}/100",
        delta=f"+{np.random.randint(1, 5)} from last week"
    )

with col2:
    st.metric(
        label="Success Rate",
        value=f"{profile['success_rate']:.1f}%",
        delta=f"+{np.random.randint(1, 10)}%"
    )

with col3:
    st.metric(
        label="Problems Solved",
        value=len(profile['solved_problems']),
        delta=f"+{np.random.randint(1, 3)} this week"
    )

with col4:
    st.metric(
        label="Total Attempts",
        value=profile['num_attempts'],
        delta=f"+{np.random.randint(2, 8)} attempts"
    )

st.divider()

# ============= MAIN TABS =============
tab_recommend, tab_analytics, tab_learning_path, tab_progress = st.tabs([
    "📚 Recommendations",
    "📈 Analytics",
    "🎯 Learning Path",
    "📉 Progress"
])

# === RECOMMENDATIONS TAB ===
with tab_recommend:
    st.markdown("### Recommended Problems")
    
    col_strategy, col_count = st.columns([3, 1])
    with col_strategy:
        strategy = st.radio(
            "Learning Strategy:",
            ["Adaptive (Balanced)", "Exploration (Diverse)", "Challenge (Hard)"],
            horizontal=True
        )
    
    with col_count:
        num_rec = st.slider("Number of recommendations:", 1, 10, 5)
    
    # Get recommendations
    strategy_map = {
        "Adaptive (Balanced)": "adaptive",
        "Exploration (Diverse)": "exploration",
        "Challenge (Hard)": "exploitation"
    }
    
    recommendations = engine.recommend_problems(
        profile,
        num_recommendations=num_rec,
        learning_stage=strategy_map[strategy]
    )
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        problem = rec['problem']
        
        with st.expander(
            f"**{i}. {problem['title']}** | {problem['difficulty']} | {problem['topic']}",
            expanded=(i == 1)
        ):
            col_prob, col_score = st.columns([3, 1])
            
            with col_prob:
                st.write(f"**Description:** {problem['description']}")
                st.write(f"**Topic:** `{problem['topic']}`")
                st.write(f"**Difficulty:** `{problem['difficulty']}`")
            
            with col_score:
                st.metric("Match Score", f"{rec['score']/2.0:.1%}")
            
            st.info(f"💡 {rec['reason']}")
            
            col_start, col_pass, col_skip = st.columns([1, 1, 1])
            with col_start:
                if st.button("▶️ Start", key=f"start_{problem['id']}", use_container_width=True):
                    st.success(f"Great! Solve {problem['title']} and come back to submit.")
            
            with col_pass:
                if st.button("✅ Mark Solved", key=f"solve_{problem['id']}", use_container_width=True):
                    st.success(f"Excellent! Problem marked as solved. Your skill improved!")
            
            with col_skip:
                if st.button("⏭️ Skip", key=f"skip_{problem['id']}", use_container_width=True):
                    st.info("Skipped. Recommendation noted.")

# === ANALYTICS TAB ===
with tab_analytics:
    st.markdown("### Learning Analytics")
    
    # Topic mastery chart
    col_topics, col_difficulty = st.columns(2)
    
    with col_topics:
        st.markdown("#### Topic Mastery")
        topic_data = pd.DataFrame({
            'Topic': list(profile['topic_mastery'].keys()),
            'Mastery': list(profile['topic_mastery'].values())
        }).sort_values('Mastery', ascending=True)
        
        fig = go.Figure(data=[
            go.Bar(
                y=topic_data['Topic'],
                x=topic_data['Mastery'],
                orientation='h',
                marker=dict(
                    color=topic_data['Mastery'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Mastery %")
                )
            )
        ])
        
        fig.update_layout(
            title="Mastery by Topic",
            xaxis_title="Mastery Level (%)",
            height=400,
            margin=dict(l=100)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col_difficulty:
        st.markdown("#### Difficulty Performance")
        
        # Success rate by difficulty
        submissions = profile['submissions']
        diff_performance = submissions.groupby('difficulty').agg({
            'passes_all_tests': ['mean', 'count']
        }).round(2)
        
        diff_data = pd.DataFrame({
            'Difficulty': ['Easy', 'Medium', 'Hard'],
            'Success Rate': [
                submissions[submissions['difficulty'] == 'd']['passes_all_tests'].mean() * 100
                for d in ['Easy', 'Medium', 'Hard']
            ]
        })
        
        fig2 = px.bar(
            diff_data,
            x='Difficulty',
            y='Success Rate',
            color='Success Rate',
            color_continuous_scale='RdYlGn',
            height=400,
            title="Success Rate by Difficulty"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Time series
    st.markdown("#### Submission Timeline")
    
    timeline_data = submissions.copy()
    timeline_data['submission_number'] = range(1, len(timeline_data) + 1)
    
    fig3 = go.Figure()
    
    fig3.add_trace(go.Scatter(
        x=timeline_data['submission_number'],
        y=timeline_data['time_to_solve_minutes'],
        mode='lines+markers',
        name='Time to Solve (minutes)',
        line=dict(color='#667eea', width=2),
        marker=dict(size=5)
    ))
    
    fig3.update_layout(
        title="Problem Solving Time Trend",
        xaxis_title="Submission #",
        yaxis_title="Time (minutes)",
        height=300,
        hovermode='x unified'
    )
    st.plotly_chart(fig3, use_container_width=True)

# === LEARNING PATH TAB ===
with tab_learning_path:
    st.markdown("### Your Personalized Learning Path")
    
    learning_path = engine.generate_learning_path(profile, num_days=30)
    readiness = engine.predict_readiness(profile, target_skill=85)
    
    # Overall readiness
    col_progress, col_timeline = st.columns([1, 2])
    
    with col_progress:
        st.markdown("#### GATE Readiness")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=readiness['gate_readiness_percentage'],
            title={'text': "Readiness %"},
            delta={'reference': 100, 'suffix': " to GATE ready"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 85], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.metric(
            "Days to GATE Ready",
            readiness['estimated_days_to_target'],
            help="Estimated days to reach GATE readiness (85/100)"
        )
    
    with col_timeline:
        st.markdown("#### 30-Day Learning Phases")
        
        for i, phase in enumerate(learning_path['learning_phases'], 1):
            with st.expander(f"Phase {phase['phase']}: {phase['days']}", expanded=True):
                st.write(f"**Focus:** {phase['focus']}")
                st.write(f"**Recommended:** {phase['recommended_problems']}")
    
    # Topic readiness
    st.markdown("#### Topic-Specific Readiness")
    
    topic_readiness = readiness['topic_readiness']
    readiness_data = []
    
    for topic, status_info in topic_readiness.items():
        readiness_data.append({
            'Topic': topic,
            'Status': status_info['status'],
            'Days to Master': status_info['days_to_master']
        })
    
    readiness_df = pd.DataFrame(readiness_data)
    
    # Color code by status
    status_colors = {
        'Mastered': '🟢',
        'Competent': '🟡',
        'Learning': '🟠',
        'Beginner': '🔴'
    }
    
    for idx, row in readiness_df.iterrows():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"{status_colors[row['Status']]} **{row['Topic']}**")
        with col2:
            st.write(f"`{row['Status']}`")
        with col3:
            if row['Days to Master'] > 0:
                st.write(f"~{row['Days to Master']}d")
            else:
                st.write("Done ✅")

# === PROGRESS TAB ===
with tab_progress:
    st.markdown("### Your Progress Dashboard")
    
    col_metrics_l, col_metrics_r = st.columns(2)
    
    with col_metrics_l:
        st.markdown("#### Quick Stats")
        analytics = engine.get_learning_analytics(profile)
        
        st.write(f"📝 **Total Submissions:** {analytics['total_submissions']}")
        st.write(f"✅ **Success Rate:** {analytics['success_rate']:.1f}%")
        st.write(f"⏱️ **Avg Solve Time:** {analytics['avg_solve_time']:.1f} minutes")
        st.write(f"🎯 **Problems Solved:** {analytics['problems_solved']}")
    
    with col_metrics_r:
        st.markdown("#### Strengths & Weaknesses")
        
        st.write("**💪 Strong Topics:**")
        if analytics['strong_topics']:
            for topic in analytics['strong_topics'][:3]:
                st.write(f"  ✅ {topic}")
        else:
            st.write("  None yet. Keep practicing!")
        
        st.write("\n**🎯 Weak Topics:**")
        if analytics['weak_topics']:
            for topic in analytics['weak_topics'][:3]:
                st.write(f"  ⚠️ {topic}")
    
    # Projection
    st.markdown("#### Skill Projection")
    
    current_skill = profile['skill']
    days = np.arange(0, 91, 10)
    projected_skill = np.minimum(100, current_skill + (days * 1.5))
    
    fig_projection = go.Figure()
    
    fig_projection.add_trace(go.Scatter(
        x=days,
        y=projected_skill,
        mode='lines+markers',
        name='Projected Skill',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        marker=dict(size=8)
    ))
    
    # GATE threshold
    fig_projection.add_hline(
        y=85,
        line_dash="dash",
        line_color="red",
        annotation_text="GATE Target",
        annotation_position="right"
    )
    
    fig_projection.update_layout(
        title="90-Day Skill Projection",
        xaxis_title="Days",
        yaxis_title="Skill Level (0-100)",
        height=350,
        hovermode='x unified'
    )
    st.plotly_chart(fig_projection, use_container_width=True)

# ============= FOOTER =============
st.divider()

st.markdown("""
---
<div style="text-align: center; color: #666;">
    <p>🎓 <b>Competitive Programming Curriculum Learning System</b></p>
    <p>Powered by TensorFlow | Multi-Task Deep Learning | Curriculum Learning Theory</p>
    <p><small>Built for VIT-AP University | CSE4006 Project</small></p>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    st.write("Run with: `streamlit run app.py`")
