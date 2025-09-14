# Create the final deployment-ready Streamlit app that uses the advanced agent
final_streamlit_app = '''import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

# Import our advanced agent
from advanced_agent import AdvancedQuickQueryAgent, format_response_for_display, calculate_usage_cost

# Page configuration
st.set_page_config(
    page_title="QuickQuery - Advanced AI Student Helpdesk",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'agent' not in st.session_state:
        st.session_state.agent = AdvancedQuickQueryAgent()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'api_configured' not in st.session_state:
        st.session_state.api_configured = False
    
    if 'show_analytics' not in st.session_state:
        st.session_state.show_analytics = False
    
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = []

initialize_session_state()

# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 QuickQuery: Advanced AI Student Helpdesk</h1>
    <p>Powered by OpenAI GPT-3.5/4 | OpenAI × NxtWave Buildathon 2025</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Key Configuration
    st.markdown("#### OpenAI API Setup")
    api_key_input = st.text_input(
        "Enter OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    
    model_choice = st.selectbox(
        "Select Model",
        options=["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo-16k"],
        index=0,
        help="GPT-4 provides better responses but costs more"
    )
    
    if api_key_input and not st.session_state.api_configured:
        if st.session_state.agent.setup_openai(api_key_input):
            st.session_state.agent.model = model_choice
            st.session_state.api_configured = True
            st.success("✅ OpenAI API Connected!")
            st.rerun()
        else:
            st.error("❌ Invalid API Key or Connection Failed")
    
    # Status indicator
    status_class = "status-online" if st.session_state.api_configured else "status-offline"
    status_text = "API Connected" if st.session_state.api_configured else "API Not Connected"
    
    st.markdown(f"""
    <div style="margin: 1rem 0;">
        <span class="status-indicator {status_class}"></span>
        <strong>{status_text}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.api_configured:
        st.warning("🔄 Using fallback responses. Add API key for enhanced AI responses.")
    
    st.markdown("---")
    
    # Quick Action Buttons
    st.markdown("### 🎯 Quick Queries")
    
    sample_categories = {
        "📚 Academic": [
            "When is the next exam?",
            "Show me AI/ML syllabus",
            "Assignment deadlines?",
            "Class timetable?"
        ],
        "🏢 Campus Facilities": [
            "Library timings?",
            "Canteen timings?", 
            "Gym facilities?",
            "Hostel information?"
        ],
        "🎉 Events & Services": [
            "Upcoming events?",
            "Placement information?",
            "Contact academic office",
            "Medical center details?"
        ]
    }
    
    for category, queries in sample_categories.items():
        with st.expander(category):
            for query in queries:
                if st.button(query, key=f"sample_{query}", use_container_width=True):
                    with st.spinner("Processing..."):
                        result = st.session_state.agent.process_query(
                            query, 
                            st.session_state.conversation_context
                        )
                        st.session_state.chat_history.append(result)
                        
                        # Update conversation context
                        st.session_state.conversation_context.append({
                            "role": "user", "content": query
                        })
                        st.session_state.conversation_context.append({
                            "role": "assistant", "content": result.ai_response
                        })
                        
                        # Keep context manageable
                        if len(st.session_state.conversation_context) > 10:
                            st.session_state.conversation_context = st.session_state.conversation_context[-10:]
                    
                    st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.markdown("### 🛠️ Settings")
    
    max_history = st.slider("Chat History Limit", 5, 50, 20)
    show_query_details = st.checkbox("Show Query Details", value=False)
    show_timestamps = st.checkbox("Show Timestamps", value=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.show_analytics = not st.session_state.show_analytics
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_context = []
            st.session_state.agent.conversation_history = []
            st.rerun()

# Main Interface
col1, col2 = st.columns([7, 3])

with col1:
    st.markdown("### 💬 Chat Interface")
    
    # Chat display area
    chat_container = st.container()
    
    with chat_container:
        if st.session_state.chat_history:
            for i, result in enumerate(reversed(st.session_state.chat_history[-max_history:])):
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑 You</strong> {f"({result.timestamp})" if show_timestamps else ""}
                    <br>{result.user_input}
                </div>
                """, unsafe_allow_html=True)
                
                # Bot response  
                formatted_response = format_response_for_display(result.ai_response)
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 QuickQuery</strong> {f"({result.timestamp})" if show_timestamps else ""}
                    <br>{formatted_response}
                </div>
                """, unsafe_allow_html=True)
                
                # Query details
                if show_query_details:
                    detail_cols = st.columns(4)
                    with detail_cols[0]:
                        st.caption(f"Type: {result.query_type}")
                    with detail_cols[1]:
                        st.caption(f"Confidence: {result.confidence:.2f}")
                    with detail_cols[2]:
                        st.caption(f"Time: {result.response_time_ms}ms")
                    with detail_cols[3]:
                        if result.tokens_used > 0:
                            cost = calculate_usage_cost(result.tokens_used, st.session_state.agent.model)
                            st.caption(f"Cost: ${cost:.4f}")
                
                st.markdown("---")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666;">
                <h3>👋 Welcome to QuickQuery!</h3>
                <p>I'm your AI-powered campus assistant. Ask me anything about:</p>
                <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                    <li>📚 Exam schedules and academic information</li>
                    <li>📝 Assignment deadlines and syllabus</li> 
                    <li>🏢 Campus facilities and services</li>
                    <li>🎉 Events and placement opportunities</li>
                    <li>📞 Contact information and support</li>
                </ul>
                <p><em>Try the quick queries in the sidebar or type your question below!</em></p>
            </div>
            """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Live Statistics")
    
    # Get current analytics
    if st.session_state.chat_history:
        analytics = st.session_state.agent.get_analytics_summary()
        
        # Key metrics
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Total Queries", analytics['total_queries'])
            st.metric("Success Rate", f"{analytics['success_rate']:.1f}%")
        
        with metric_col2:
            st.metric("Avg Confidence", f"{analytics['average_confidence']:.2f}")
            st.metric("Avg Response", f"{analytics['average_response_time']:.0f}ms")
        
        if analytics['total_tokens'] > 0:
            total_cost = calculate_usage_cost(analytics['total_tokens'], st.session_state.agent.model)
            st.metric("Total Cost", f"${total_cost:.4f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Query type distribution (mini chart)
        if analytics.get('query_distribution'):
            st.markdown("#### Query Types")
            query_types = list(analytics['query_distribution'].keys())[:5]
            query_counts = [analytics['query_distribution'][qt] for qt in query_types]
            
            fig = px.bar(
                x=query_counts,
                y=query_types,
                orientation='h',
                height=200
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                yaxis={'tickfont': {'size': 10}}
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    else:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Status", "Ready")
        st.metric("Mode", "Demo Mode" if not st.session_state.api_configured else "AI Mode")
        st.metric("Uptime", "Just Started")
        st.markdown('</div>', unsafe_allow_html=True)

# Input Section
st.markdown("---")
st.markdown("### ✍️ Ask QuickQuery")

input_col1, input_col2 = st.columns([5, 1])

with input_col1:
    user_input = st.text_area(
        "",
        placeholder="Type your question here... (e.g., When is the next exam? What are library timings? Tell me about upcoming events?)",
        height=100,
        key="user_input",
        label_visibility="collapsed"
    )

with input_col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
    
    if st.button("🚀 Send", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("🤖 QuickQuery is thinking..."):
                result = st.session_state.agent.process_query(
                    user_input, 
                    st.session_state.conversation_context
                )
                st.session_state.chat_history.append(result)
                
                # Update conversation context
                st.session_state.conversation_context.append({
                    "role": "user", "content": user_input
                })
                st.session_state.conversation_context.append({
                    "role": "assistant", "content": result.ai_response
                })
                
                # Keep context manageable
                if len(st.session_state.conversation_context) > 10:
                    st.session_state.conversation_context = st.session_state.conversation_context[-10:]
            
            st.rerun()
        else:
            st.warning("Please enter a question!")

# Analytics Dashboard (if enabled)
if st.session_state.show_analytics and st.session_state.chat_history:
    st.markdown("---")
    st.markdown("## 📈 Advanced Analytics Dashboard")
    
    analytics = st.session_state.agent.get_analytics_summary()
    
    # Overview metrics
    metric_cols = st.columns(5)
    with metric_cols[0]:
        st.metric("Total Queries", analytics['total_queries'])
    with metric_cols[1]:
        st.metric("Success Rate", f"{analytics['success_rate']:.1f}%")
    with metric_cols[2]:
        st.metric("Avg Confidence", f"{analytics['average_confidence']:.2f}")
    with metric_cols[3]:
        st.metric("Avg Response Time", f"{analytics['average_response_time']:.0f}ms")
    with metric_cols[4]:
        if analytics['total_tokens'] > 0:
            total_cost = calculate_usage_cost(analytics['total_tokens'], st.session_state.agent.model)
            st.metric("Session Cost", f"${total_cost:.4f}")
        else:
            st.metric("Mode", "Free Tier")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Query type distribution
        if analytics.get('query_distribution'):
            st.subheader("Query Type Distribution")
            fig = px.pie(
                values=list(analytics['query_distribution'].values()),
                names=list(analytics['query_distribution'].keys()),
                title="Distribution by Query Type"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        # Response time trend
        st.subheader("Response Time Trend")
        if len(st.session_state.chat_history) > 1:
            response_times = [q.response_time_ms for q in st.session_state.chat_history]
            query_numbers = list(range(1, len(response_times) + 1))
            
            fig = px.line(
                x=query_numbers,
                y=response_times,
                title="Response Time Over Queries",
                labels={'x': 'Query Number', 'y': 'Response Time (ms)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more queries to show trend")
    
    # Export functionality
    if st.button("📥 Export Conversation Data"):
        export_data = st.session_state.agent.export_conversation_data()
        df = pd.DataFrame(export_data)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"quickquery_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;'>
    <h4>🏆 OpenAI × NxtWave Buildathon 2025 Submission</h4>
    <p><strong>QuickQuery: Advanced AI Student Helpdesk</strong></p>
    <p>🤖 Powered by OpenAI GPT | 📊 Advanced Analytics | 🎯 Smart Query Classification</p>
    <p>✨ <em>Built with Streamlit, OpenAI API, and lots of ❤️</em></p>
    
    <details>
        <summary><strong>🔧 Technical Features</strong></summary>
        <ul style="text-align: left; max-width: 600px; margin: 1rem auto;">
            <li>🧠 Advanced AI query classification with confidence scoring</li>
            <li>💬 Context-aware conversation handling</li>
            <li>📊 Real-time analytics and usage monitoring</li>
            <li>🔄 Intelligent fallback responses</li>
            <li>💰 API usage cost tracking</li>
            <li>📱 Mobile-responsive design</li>
            <li>📈 Interactive charts and visualizations</li>
            <li>💾 Data export capabilities</li>
        </ul>
    </details>
</div>
""", unsafe_allow_html=True)
'''

# Save the final enhanced app
with open('quickquery_final_app.py', 'w') as f:
    f.write(final_streamlit_app)

# Create final requirements file
final_requirements = """streamlit>=1.28.0
openai>=0.28.0
pandas>=2.0.0
plotly>=5.15.0
python-dotenv>=1.0.0
requests>=2.31.0
numpy>=1.24.0
"""

with open('final_requirements.txt', 'w') as f:
    f.write(final_requirements)

print("✅ COMPLETE ENHANCED PROJECT READY!")
print("📁 Files created:")
print("  - quickquery_final_app.py (main Streamlit app)")
print("  - advanced_agent.py (AI agent implementation)")
print("  - final_requirements.txt (dependencies)")
print("  - config.py (configuration management)")
print("  - .env.template (environment setup)")
print()
print("🚀 FEATURES INCLUDED:")
print("  ✅ Real OpenAI API integration with GPT-3.5/4")
print("  ✅ Advanced query classification with confidence")
print("  ✅ Interactive analytics dashboard")
print("  ✅ Professional UI with custom styling")
print("  ✅ Cost tracking and usage monitoring")
print("  ✅ Conversation context awareness")
print("  ✅ Mobile-responsive design")
print("  ✅ Data export capabilities")
print("  ✅ Error handling and fallbacks")
print("  ✅ Production-ready code structure")
print()
print("🌐 DEPLOYMENT READY:")
print("  - Streamlit Cloud: Use quickquery_final_app.py")
print("  - Heroku/Railway: All files included")
print("  - Local development: python -m streamlit run quickquery_final_app.py")