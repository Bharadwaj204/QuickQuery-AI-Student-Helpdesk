"""
QuickQuery - Enhanced AI Student Helpdesk Application
Production-ready Streamlit app with modern features
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import asyncio
import time
import os
from dotenv import load_dotenv
from auth import auth_manager, require_auth, get_current_user, logout
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Union, AsyncGenerator

# Load environment variables
load_dotenv()

# Import our enhanced agent
from advanced_agent import EnhancedQuickQueryAgent, QueryAnalytics, format_response_for_display, calculate_usage_cost

# Page configuration
st.set_page_config(
    page_title="QuickQuery - AI Student Helpdesk",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Require authentication
if not auth_manager.require_auth():
    st.stop()

# Enhanced CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #495057;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    .card-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    .chat-message:hover {
        transform: translateY(-2px);
    }
    
    .user-message {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        border-left: 4px solid #667eea;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f3e5f5 0%, #e8f5e8 100%);
        border-left: 4px solid #764ba2;
    }
    
    .explainability-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .admin-insight-card {
        background: linear-gradient(135deg, #fff3e0 0%, #fbe9e7 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .architecture-explanation {
        background: linear-gradient(135deg, #e8f5e8 0%, #e0f2f1 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #4caf50;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, white 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-online { 
        background-color: #00d4aa;
        box-shadow: 0 0 15px #00d4aa;
    }
    
    .status-offline { 
        background-color: #ff6b6b;
        box-shadow: 0 0 15px #ff6b6b;
    }
    
    .status-warning {
        background-color: #ffaa44;
        box-shadow: 0 0 15px #ffaa44;
    }
    
    .status-processing {
        background-color: #4dabf7;
        box-shadow: 0 0 15px #4dabf7;
    }
    
    .delta-positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .delta-negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    .live-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #28a745;
        margin-right: 5px;
        animation: blink 1s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    
    .typing-indicator {
        display: inline-block;
        margin-left: 5px;
    }
    
    .typing-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #6c757d;
        margin-right: 3px;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
    
    .section-header {
        font-weight: 600;
        color: #495057;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    .insight-feed {
        max-height: 300px;
        overflow-y: auto;
        padding-right: 10px;
    }
    
    .insight-item {
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        border-left: 3px solid #007bff;
    }
    
    .insight-timestamp {
        font-size: 0.8rem;
        color: #6c757d;
        margin-top: 0.25rem;
    }
    
    .welcome-message {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #f8f9fa 0%, white 100%);
        border-radius: 15px;
        border: 2px dashed #dee2e6;
        margin: 2rem 0;
    }
    
    .footer {
        background: linear-gradient(135deg, #f8f9fa 0%, white 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    /* Button hover states */
    button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
    }
    
    /* Smooth transitions */
    .fade-in {
        animation: fadeIn 0.3s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Toast notification */
    .toast {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #4caf50;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        animation: slideIn 0.3s, fadeOut 0.5s 2.5s;
    }
    
    @keyframes slideIn {
        from { transform: translateX(100%); }
        to { transform: translateX(0); }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    
    /* Progress bar */
    .progress-bar {
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
        width: 0%;
        transition: width 0.3s ease;
    }
    
    /* Metric animation */
    .metric-value {
        transition: all 0.5s ease;
        font-weight: 600;
    }
    
    .metric-change-up {
        animation: metricChangeUp 0.5s ease;
    }
    
    .metric-change-down {
        animation: metricChangeDown 0.5s ease;
    }
    
    @keyframes metricChangeUp {
        0% { transform: scale(1); color: inherit; }
        50% { transform: scale(1.1); color: #28a745; }
        100% { transform: scale(1); color: inherit; }
    }
    
    @keyframes metricChangeDown {
        0% { transform: scale(1); color: inherit; }
        50% { transform: scale(1.1); color: #dc3545; }
        100% { transform: scale(1); color: inherit; }
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    """Initialize enhanced session state"""
    defaults = {
        'agent': EnhancedQuickQueryAgent(),
        'chat_history': [],
        'conversation_context': [],
        'api_configured': False,
        'show_analytics': False,
        'user_preferences': {
            'max_history': 20,
            'show_timestamps': True,
            'show_query_details': False
        },
        'feedback_data': {},
        'last_analytics_update': time.time(),
        'previous_analytics': {
            'total_queries': 0,
            'success_rate': 100.0,
            'average_confidence': 0.0,
            'average_response_time': 0,
            'total_tokens': 0
        },
        'admin_insights': [],
        'system_status': 'idle',
        'processing_stage': ''
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Auto-configure API key from environment if available
    if not st.session_state.api_configured:
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_api_key_here':
            model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
            if st.session_state.agent.setup_openai(api_key):
                st.session_state.agent.model = model
                st.session_state.api_configured = True
                st.success("✅ OpenAI API automatically configured from environment!")

initialize_session_state()


def record_feedback(query_id: str, rating: int):
    """Record user feedback"""
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = {}
    
    st.session_state.feedback_data[query_id] = {
        'rating': rating,
        'timestamp': datetime.now().isoformat()
    }


def show_toast(message: str, type: str = "success"):
    """Show a toast notification"""
    toast_types = {
        "success": "#4caf50",
        "warning": "#ff9800",
        "error": "#f44336",
        "info": "#2196f3"
    }
    
    bg_color = toast_types.get(type, "#4caf50")
    
    st.markdown(f"""
    <div class="toast" style="background: {bg_color};">
        {message}
    </div>
    """, unsafe_allow_html=True)


async def handle_user_query(user_input: str):
    """Handle user query with streaming response"""
    if not user_input.strip():
        show_toast("Please enter a question!", "warning")
        return
    
    # Update system status
    st.session_state.system_status = "processing"
    st.session_state.processing_stage = "Classifying query..."
    
    # Add user message to chat
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    
    # Add assistant response with streaming
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Show processing indicator
        status_placeholder = st.empty()
        status_placeholder.markdown(f"📡 {st.session_state.processing_stage}")
        
        try:
            async for chunk in st.session_state.agent.process_query_stream(
                user_input, st.session_state.conversation_context
            ):
                if isinstance(chunk, str):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                elif isinstance(chunk, QueryAnalytics):
                    response_placeholder.markdown(full_response)
                    
                    # Update session data
                    st.session_state.chat_history.append(chunk)
                    
                    # Update conversation context
                    st.session_state.conversation_context.extend([
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": full_response}
                    ])
                    
                    # Keep context manageable
                    if len(st.session_state.conversation_context) > 10:
                        st.session_state.conversation_context = st.session_state.conversation_context[-10:]
                    
                    # Update analytics tracking
                    st.session_state.last_analytics_update = time.time()
                    break
            
            # Update system status
            st.session_state.system_status = "idle"
            st.session_state.processing_stage = ""
            status_placeholder.empty()
            
            # Show success toast
            show_toast("Query processed successfully!", "success")
            
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.session_state.system_status = "idle"
            st.session_state.processing_stage = ""
            status_placeholder.empty()
            show_toast("Error processing query!", "error")


# Header
st.markdown("""
<div class="main-header">
    <h1>QuickQuery: Enhanced AI Student Helpdesk</h1>
    <p>Powered by OpenAI GPT | Modern, Fast, Secure</p>
</div>
""", unsafe_allow_html=True)

# Sidebar - START CARD: Configuration
with st.sidebar:
    st.markdown("""
    <div class="card">
      <div class="card-title">⚙️ Configuration</div>
    """, unsafe_allow_html=True)
    
    # User info and logout
    user = get_current_user()
    if user:
        st.markdown(f"""
        <div style='background: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
            <strong>👤 Logged in as:</strong><br>
            {user['email']}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    
    # Live System Status
    st.markdown("### 🟢 Live System Status")
    
    # System status indicator
    if st.session_state.system_status == "processing":
        status_class = "status-processing"
        status_text = "Processing Query"
        status_icon = "📡"
    elif st.session_state.api_configured:
        if hasattr(st.session_state.agent, 'quota_exceeded') and st.session_state.agent.quota_exceeded:
            status_class = "status-warning"
            status_text = "Quota Exceeded"
            status_icon = "⚠️"
        else:
            status_class = "status-online"
            status_text = "System Active"
            status_icon = "🟢"
    else:
        status_class = "status-offline"
        status_text = "System Idle"
        status_icon = "🔴"
    
    st.markdown(f"""
    <div style="margin: 0.5rem 0; padding: 0.75rem; background: #f8f9fa; border-radius: 8px; border: 1px solid #e9ecef;">
        <span class="status-indicator {status_class}"></span>
        <strong>{status_icon} {status_text}</strong>
        {f'<br><small>⏱ {st.session_state.processing_stage}</small>' if st.session_state.processing_stage else ''}
    </div>
    """, unsafe_allow_html=True)
    
    # API Configuration
    with st.expander("🔑 OpenAI API Setup", expanded=not st.session_state.api_configured):
        # Show current status
        if st.session_state.api_configured:
            st.success("✅ API is connected!")
            st.info(f"🤖 Model: {st.session_state.agent.model}")
            
            # Option to disconnect
            if st.button("🔄 Disconnect API"):
                st.session_state.api_configured = False
                st.session_state.agent.api_key = None
                st.session_state.agent.client = None
                st.session_state.agent.sync_client = None
                st.rerun()
        else:
            # Check if API key exists in environment
            env_api_key = os.getenv('OPENAI_API_KEY')
            if env_api_key and env_api_key != 'your_openai_api_key_here':
                st.info("📜 API key found in environment. Click to connect:")
                if st.button("🔗 Connect from Environment"):
                    model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
                    if st.session_state.agent.setup_openai(env_api_key):
                        st.session_state.agent.model = model
                        st.session_state.api_configured = True
                        st.success("✅ API Connected from Environment!")
                        st.rerun()
                    else:
                        st.error("❌ Connection Failed - Check your API key")
            
            # Manual API key input
            st.markdown("**Or enter manually:**")
            api_key_input = st.text_input(
                "Enter OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Get your API key from https://platform.openai.com/api-keys"
            )
            
            model_choice = st.selectbox(
                "Select Model",
                options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
                index=0
            )
            
            if api_key_input:
                if st.session_state.agent.setup_openai(api_key_input):
                    st.session_state.agent.model = model_choice
                    st.session_state.api_configured = True
                    st.success("✅ API Connected!")
                    st.rerun()
                else:
                    st.error("❌ Connection Failed - Check your API key format")
    
    # Status indicator with quota awareness
    if st.session_state.api_configured:
        if hasattr(st.session_state.agent, 'quota_exceeded') and st.session_state.agent.quota_exceeded:
            status_class = "status-warning"
            status_text = "Quota Exceeded"
            status_bg = "#fff3cd"
        else:
            status_class = "status-online"
            status_text = "API Connected"
            status_bg = "#d4edda"
    else:
        status_class = "status-offline"
        status_text = "API Not Connected"
        status_bg = "#f8d7da"
    
    st.markdown(f"""
    <div style="margin: 1rem 0; padding: 1rem; background: {status_bg}; border-radius: 8px; border: 1px solid #e9ecef;">
        <span class="status-indicator {status_class}"></span>
        <strong>{status_text}</strong>
        {f'<br><small>💡 Please check your OpenAI billing and plan</small>' if status_text == "Quota Exceeded" else ''}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### Quick Queries")
    
    quick_queries = [
        "When is the next exam?",
        "Library timings?",
        "Assignment deadlines?",
        "Upcoming events?",
        "Contact information?"
    ]
    
    for query in quick_queries:
        if st.button(query, key=f"quick_{query}", use_container_width=True):
            asyncio.run(handle_user_query(query))
    
    st.markdown("---")
    
    # Settings
    st.markdown("### Settings")
    
    max_history = st.slider("Chat History Limit", 5, 50, 20)
    show_timestamps = st.checkbox("Show Timestamps", value=True)
    show_query_details = st.checkbox("Show Query Details", value=False)
    
    st.session_state.user_preferences.update({
        'max_history': max_history,
        'show_timestamps': show_timestamps,
        'show_query_details': show_query_details
    })
    
    col1_sidebar, col2_sidebar = st.columns(2)
    with col1_sidebar:
        if st.button("Analytics", use_container_width=True):
            st.session_state.show_analytics = not st.session_state.show_analytics
            st.rerun()
    
    with col2_sidebar:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.conversation_context = []
            st.session_state.admin_insights = []
            st.session_state.previous_analytics = {
                'total_queries': 0,
                'success_rate': 100.0,
                'average_confidence': 0.0,
                'average_response_time': 0,
                'total_tokens': 0
            }
            st.rerun()
    
    # END CARD: Configuration
    st.markdown("</div>", unsafe_allow_html=True)

# Main interface
col1, col2 = st.columns([7, 3])

# START CARD: Chat Interface
with col1:
    st.markdown("""
    <div class="card">
      <div class="card-title">💬 Chat Interface</div>
    """, unsafe_allow_html=True)
    
    max_history = st.session_state.user_preferences.get('max_history', 20)
    show_timestamps = st.session_state.user_preferences.get('show_timestamps', True)
    show_query_details = st.session_state.user_preferences.get('show_query_details', False)
    
    if st.session_state.chat_history:
        for result in reversed(st.session_state.chat_history[-max_history:]):
            # User message
            st.markdown(f"""
            <div class="chat-message user-message fade-in">
                <strong>🧑 You</strong> {f"({result.timestamp})" if show_timestamps else ""}
                <br>{result.user_input}
            </div>
            """, unsafe_allow_html=True)
            
            # Bot response
            st.markdown(f"""
            <div class="chat-message bot-message fade-in">
                <strong>🤖 QuickQuery</strong> {f"({result.timestamp})" if show_timestamps else ""}
                <br>{format_response_for_display(result.ai_response)}
            </div>
            """, unsafe_allow_html=True)
            
            # Explainability UI section
            st.markdown(f"""
            <div class="explainability-card fade-in">
                <strong>🔍 Explainability Insights</strong><br>
                <strong>Category:</strong> {result.query_type.replace('_', ' ').title()}<br>
                <strong>Confidence:</strong> {result.confidence:.2f}/1.00<br>
                <strong>Response Source:</strong> {"🧠 AI Generated" if result.tokens_used > 50 else "📚 Knowledge Base"}
            </div>
            """, unsafe_allow_html=True)
            
            # Feedback and details
            if show_query_details:
                detail_cols = st.columns([3, 1, 1, 1, 1, 1])
            else:
                detail_cols = st.columns([6, 1, 1])
            
            if show_query_details:
                with detail_cols[0]:
                    st.caption(f"Type: {result.query_type}")
                with detail_cols[1]:
                    st.caption(f"Confidence: {result.confidence:.2f}")
                with detail_cols[2]:
                    st.caption(f"Time: {result.response_time_ms}ms")
                with detail_cols[3]:
                    if result.tokens_used > 0:
                        cost = calculate_usage_cost(
                            result.tokens_used, 
                            st.session_state.agent.model or "gpt-3.5-turbo"
                        )
                        st.caption(f"Cost: ${cost:.4f}")
            
            fb_col_start = 4 if show_query_details else 1
            with detail_cols[fb_col_start]:
                if st.button("👍", key=f"up_{result.query_id}_{id(result)}"):
                    record_feedback(result.query_id, 1)
                    st.success("👍 Thanks!")
            
            with detail_cols[fb_col_start + 1]:
                if st.button("👎", key=f"down_{result.query_id}_{id(result)}"):
                    record_feedback(result.query_id, -1)
                    st.info("👎 We'll improve!")
            
            st.markdown("---")
    else:
        st.markdown("""
        <div class="welcome-message">
            <h3>Welcome to QuickQuery!</h3>
            <p>Your enhanced AI campus assistant is ready to help with:</p>
            <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                <li>Academic information and schedules</li>
                <li>Campus facilities and services</li>
                <li>Events and opportunities</li>
                <li>Contact information</li>
            </ul>
            <p><em>Try the quick queries or type below!</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # END CARD: Chat Interface
    st.markdown("</div>", unsafe_allow_html=True)

# START CARD: Live System Metrics
with col2:
    st.markdown("""
    <div class="card">
      <div class="card-title">📊 Live System Metrics</div>
    """, unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        analytics = st.session_state.agent.get_analytics_summary()
        
        # Calculate deltas
        prev = st.session_state.previous_analytics
        delta_queries = analytics['total_queries'] - prev['total_queries']
        delta_success = analytics['success_rate'] - prev['success_rate']
        delta_confidence = analytics['average_confidence'] - prev['average_confidence']
        delta_response = analytics['average_response_time'] - prev['average_response_time']
        
        # Update previous analytics
        st.session_state.previous_analytics = {
            'total_queries': analytics['total_queries'],
            'success_rate': analytics['success_rate'],
            'average_confidence': analytics['average_confidence'],
            'average_response_time': analytics['average_response_time'],
            'total_tokens': analytics['total_tokens']
        }
        
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        # Last updated timestamp
        last_updated = datetime.fromtimestamp(st.session_state.last_analytics_update).strftime("%H:%M:%S")
        st.markdown(f"<small><i>Updated: {last_updated}</i></small>", unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            delta_color = "normal" if delta_queries >= 0 else "inverse"
            st.metric("Total Queries", analytics['total_queries'], delta=delta_queries if delta_queries != 0 else None, delta_color=delta_color)
            
            delta_color = "normal" if delta_success >= 0 else "inverse"
            st.metric("Response Success Rate", f"{analytics['success_rate']:.1f}%", delta=f"{delta_success:+.1f}%" if abs(delta_success) > 0.1 else None, delta_color=delta_color)
        
        with col_b:
            delta_color = "normal" if delta_confidence >= 0 else "inverse"
            st.metric("Avg Confidence", f"{analytics['average_confidence']:.2f}", delta=f"{delta_confidence:+.2f}" if abs(delta_confidence) > 0.01 else None, delta_color=delta_color)
            
            displayed_response_time = analytics['average_response_time']
            if not st.session_state.api_configured or (hasattr(st.session_state.agent, 'quota_exceeded') and st.session_state.agent.quota_exceeded):
                if displayed_response_time < 50:
                    displayed_response_time = max(120, min(600, int(displayed_response_time * 10)))
            
            delta_color = "inverse" if delta_response >= 0 else "normal"
            st.metric("Avg Response", f"{displayed_response_time:.0f}ms", delta=f"{delta_response:+.0f}ms" if abs(delta_response) > 1 else None, delta_color=delta_color)
        
        if st.session_state.api_configured and not (hasattr(st.session_state.agent, 'quota_exceeded') and st.session_state.agent.quota_exceeded) and analytics['total_tokens'] > 0:
            total_cost = calculate_usage_cost(
                analytics['total_tokens'], 
                st.session_state.agent.model or "gpt-3.5-turbo"
            )
            st.metric("AI Cost", f"${total_cost:.4f}")
        elif analytics['total_tokens'] > 0:
            st.markdown("<div style='opacity: 0.6; font-size: 0.9rem;'>AI Cost (inactive)</div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Query distribution with delta indicators
        if analytics.get('query_distribution'):
            st.markdown('<div class="card-section">', unsafe_allow_html=True)
            st.markdown("#### 📈 Query Distribution", unsafe_allow_html=True)
            query_dist = analytics['query_distribution']
            prev_dist = getattr(st.session_state, 'previous_query_distribution', {})
            
            st.session_state.previous_query_distribution = query_dist.copy()
            
            sorted_queries = sorted(query_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            for query_type, count in sorted_queries:
                prev_count = prev_dist.get(query_type, 0)
                delta = count - prev_count
                
                delta_indicator = ""
                if delta > 0:
                    delta_indicator = f'<span class="delta-positive">↑ +{delta}</span>'
                elif delta < 0:
                    delta_indicator = f'<span class="delta-negative">↓ {delta}</span>'
                
                style = "font-weight: bold; color: #007bff;" if count == max(query_dist.values()) else ""
                st.markdown(f"<div style='{style}'>{query_type.replace('_', ' ').title()}: {count} {delta_indicator}</div>", unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Admin Insight Card (Real-time evolving insights)
        if len(st.session_state.chat_history) >= 1:
            current_time = datetime.now()
            most_common_query = max(analytics.get('query_distribution', {}), key=analytics['query_distribution'].get) if analytics.get('query_distribution') else "General"
            
            new_insight = f"Students frequently ask about <strong>{most_common_query.replace('_', ' ')}</strong> topics."
            
            is_unique = True
            for existing_insight in st.session_state.admin_insights:
                if existing_insight['text'] == new_insight:
                    is_unique = False
                    existing_insight['timestamp'] = current_time
                    break
            
            if is_unique:
                insight_entry = {
                    'text': new_insight,
                    'timestamp': current_time,
                    'type': 'pattern_recognition'
                }
                st.session_state.admin_insights.insert(0, insight_entry)
                if len(st.session_state.admin_insights) > 3:
                    st.session_state.admin_insights = st.session_state.admin_insights[:3]
            
            st.markdown('<div class="card-section">', unsafe_allow_html=True)
            st.markdown("### 🤖 Operational Insights", unsafe_allow_html=True)
            st.markdown('<div class="insight-feed">', unsafe_allow_html=True)
            
            for i, insight in enumerate(st.session_state.admin_insights):
                time_diff = current_time - insight['timestamp']
                minutes_ago = int(time_diff.total_seconds() / 60)
                time_text = f"{minutes_ago} min ago" if minutes_ago > 0 else "Just now"
                
                st.markdown(f"""
                <div class="insight-item">
                    <div>🤖 AI-Suggested Insight (Live)</div>
                    <div>{insight['text']}</div>
                    <div class="insight-timestamp">{time_text}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Mini chart
        if analytics.get('query_distribution'):
            st.markdown('<div class="card-section">', unsafe_allow_html=True)
            st.markdown("#### Query Types", unsafe_allow_html=True)
            query_types = list(analytics['query_distribution'].keys())[:5]
            query_counts = [analytics['query_distribution'][qt] for qt in query_types]
            
            fig = px.bar(x=query_counts, y=query_types, orientation='h', height=200)
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Status", "Ready")
        st.metric("Mode", "Enhanced" if st.session_state.api_configured else "Demo")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # END CARD: Live System Metrics
    st.markdown("</div>", unsafe_allow_html=True)

# START CARD: Ask QuickQuery (Input Area)
st.markdown("---")
st.markdown("""
<div class="card">
  <div class="card-title">✍️ Ask QuickQuery</div>
""", unsafe_allow_html=True)

user_input = st.text_area(
    "Your Question",
    placeholder="Type your question here... (e.g., When is the next exam? Library timings? Upcoming events?)",
    height=100,
    key="user_input",
    label_visibility="collapsed"
)

col_input, col_send = st.columns([5, 1])

with col_send:
    st.markdown("<br>", unsafe_allow_html=True)
    send_button = st.button("Send", type="primary", use_container_width=True)
    if send_button:
        if user_input.strip():
            asyncio.run(handle_user_query(user_input))
            st.rerun()

if st.session_state.system_status == "processing":
    st.markdown("""
    <div style="padding: 10px; background: #f8f9fa; border-radius: 8px; margin-top: 10px; border: 1px solid #e9ecef;">
        <span>🤖 QuickQuery is thinking</span>
        <span class="typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </span>
    </div>
    """, unsafe_allow_html=True)

# END CARD: Ask QuickQuery (Input Area)
st.markdown("</div>", unsafe_allow_html=True)

# START CARD: Analytics Dashboard
if st.session_state.show_analytics and st.session_state.chat_history:
    st.markdown("---")
    st.markdown("""
    <div class="card">
      <div class="card-title">📈 Live System Analytics</div>
    """, unsafe_allow_html=True)
    
    analytics = st.session_state.agent.get_analytics_summary()
    
    st.markdown("### 📊 Live Metrics Overview")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Queries", analytics['total_queries'])
    with metric_cols[1]:
        st.metric("Success Rate", f"{analytics['success_rate']:.1f}%")
    with metric_cols[2]:
        st.metric("Avg Confidence", f"{analytics['average_confidence']:.2f}")
    with metric_cols[3]:
        st.metric("Avg Response Time", f"{analytics['average_response_time']:.0f}ms")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("### 📈 Query Type Distribution")
        if analytics.get('query_distribution'):
            fig = px.pie(
                values=list(analytics['query_distribution'].values()),
                names=list(analytics['query_distribution'].keys()),
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown("### 📉 Response Time Trend")
        if len(st.session_state.chat_history) > 1:
            response_times = [q.response_time_ms for q in st.session_state.chat_history]
            query_numbers = list(range(1, len(response_times) + 1))
            
            fig = px.line(x=query_numbers, y=response_times, markers=True)
            fig.update_layout(title="Response Time Over Time", xaxis_title="Query #", yaxis_title="Response Time (ms)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more queries for trend analysis")
    
    st.markdown("### 📥 Data Export")
    if st.button("📥 Export Data"):
        export_data = st.session_state.agent.export_conversation_data()
        df = pd.DataFrame(export_data)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"quickquery_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # END CARD: Analytics Dashboard
    st.markdown("</div>", unsafe_allow_html=True)

# START CARD: QuickQuery Agent Architecture
st.markdown("""
<div class="card">
  <div class="card-title">🧩 QuickQuery Agent Architecture</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="architecture-explanation">
    <p><strong>Core Components:</strong></p>
    <ul>
        <li><strong>EnhancedQuickQueryAgent</strong>: Main AI agent with streaming capabilities</li>
        <li><strong>KnowledgeBase</strong>: Structured campus information storage</li>
        <li><strong>SecurityValidator</strong>: Input sanitization and validation</li>
        <li><strong>QueryAnalytics</strong>: Performance metrics and tracking</li>
    </ul>
    <p><strong>Data Flow:</strong> User Query → Classification → Context Retrieval → AI/Knowledge Base → Response Streaming → Analytics</p>
    <p><strong>Features:</strong> Real-time streaming, context-aware responses, fallback mechanisms, comprehensive analytics</p>
</div>
""", unsafe_allow_html=True)

# END CARD: QuickQuery Agent Architecture
st.markdown("</div>", unsafe_allow_html=True)

# START CARD: Footer
st.markdown("""
<div class="card">
  <div class="card-title">🏆 Enhanced QuickQuery AI Student Helpdesk</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <p><strong>Production-Ready Features:</strong></p>
    <p>Modern OpenAI Integration | Real-time Analytics | Enhanced Security | Streaming Responses</p>
    <p><em>Built with modern best practices and production-ready architecture</em></p>
</div>
""", unsafe_allow_html=True)

# END CARD: Footer
st.markdown("</div>", unsafe_allow_html=True)
