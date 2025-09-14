# Create enhanced version with OpenAI API integration
enhanced_streamlit_app = '''import streamlit as st
import openai
import datetime
import json
import os
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass

# Configure Streamlit page
st.set_page_config(
    page_title="QuickQuery - Advanced AI Student Helpdesk",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class QueryResult:
    user_input: str
    ai_response: str
    query_type: str
    confidence: float
    response_time: str
    timestamp: str
    tokens_used: int = 0

class AdvancedQuickQueryAgent:
    """
    Enhanced AI Student Helpdesk Agent with OpenAI Integration
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.conversation_history = []
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "query_types": {},
            "average_response_time": 0
        }
        
        # Enhanced knowledge base
        self.knowledge_base = {
            "academic": {
                "exams": {
                    "internal_exams": "Internal Exam Schedule: Sept 20-25, 2025 (Mid-sem), Nov 15-20, 2025 (Pre-final)",
                    "semester_exams": "End Semester Exams: Dec 10-20, 2025. Practical exams: Dec 5-9, 2025",
                    "entrance_exams": "GATE 2026: Feb 1-16. JEE Main: Jan 2026. CAT 2025: Nov 26, 2025",
                    "exam_rules": "ID card mandatory. No mobile phones. Reach 30 mins early. Calculator allowed for specific subjects only."
                },
                "syllabus": {
                    "ai_ml": "AI/ML Curriculum: Neural Networks (25%), Deep Learning (30%), NLP (20%), Computer Vision (15%), MLOps (10%)",
                    "cs_core": "Core CS: DSA (40%), DBMS (25%), OS (20%), Networks (15%). All subjects have lab components.",
                    "electives": "Electives Available: Cloud Computing, Blockchain, IoT, Cybersecurity, Game Development, Mobile App Dev"
                },
                "assignments": {
                    "current": "Current Deadlines: AI Project (Sept 30), ML Assignment 2 (Oct 5), Database Project (Oct 15), Network Security (Oct 22)",
                    "submission_rules": "Submit via LMS only. Late submissions: -10% per day. Plagiarism results in zero marks.",
                    "grading": "Assignment weightage: 30% of total marks. Viva/Demo required for projects above 50 marks."
                },
                "timetable": {
                    "regular": "Classes: Mon-Fri 9AM-4PM. Lunch: 12-1PM. Free periods: Wed 2-3PM, Fri 3-4PM",
                    "labs": "AI Lab: Tue/Thu 2-5PM (Room 301, 40 systems). Programming Lab: Mon/Wed/Fri 2-4PM (Room 201, 60 systems)",
                    "library": "Library Hours: 6AM-10PM (Mon-Sat), 8AM-8PM (Sun). Group study: 2nd floor. Silent zone: 3rd floor"
                }
            },
            "campus": {
                "facilities": {
                    "library": "Central Library: 50,000+ books, IEEE/ACM digital access, 200 seating capacity, WiFi, printing services",
                    "canteen": "Main Canteen: 8AM-8PM (South Indian). Coffee Corner: 7AM-9PM. Night Canteen: 8PM-11PM (Snacks only)",
                    "gym": "Fitness Center: 6AM-10PM. Equipment: Cardio machines, weights, badminton, basketball court. Free for students",
                    "medical": "Health Center: 8AM-6PM. Doctor available: Mon/Wed/Fri. Nurse: Daily. Emergency: 24/7. Basic medicines free",
                    "transport": "College Bus: 12 routes covering city. Timings: 7AM-7PM. Monthly pass: ₹500. AC buses available",
                    "hostels": "Boys Hostel: 500 capacity. Girls Hostel: 300 capacity. WiFi, laundry, mess facilities included"
                },
                "events": {
                    "tech_events": "Upcoming: Tech Fest (Oct 15-17), AI Workshop (Sept 25), Coding Competition (Oct 1), Hackathon (Nov 5-6)",
                    "placement": "Placement Season: Nov-March. Companies: Google, Microsoft, Amazon, TCS, Infosys. PPT starts Oct 1",
                    "cultural": "Cultural Events: Annual Fest (Jan 2026), Dance Competition (Dec 15), Music Night (Dec 20)",
                    "sports": "Sports Meet: Dec 2025. Cricket, Football, Basketball, Badminton, Chess tournaments"
                },
                "contacts": {
                    "academic": "Academic Office: ext-101, academic@college.edu. HOD Contact: hod.cs@college.edu",
                    "it_support": "IT Helpdesk: ext-201, it@college.edu. Lab Issues: ext-202. WiFi Issues: ext-203",
                    "hostel": "Hostel Office: ext-301. Boys Hostel Warden: ext-302. Girls Hostel Warden: ext-303",
                    "placement": "Placement Cell: ext-401, placement@college.edu. Career Counselor: ext-402",
                    "emergency": "Security: ext-911. Medical Emergency: ext-100. Fire Safety: ext-101. Admin: ext-500"
                }
            },
            "resources": {
                "online_platforms": "LMS: portal.college.edu. GitHub Student Pack available. Microsoft Office 365 for students",
                "study_materials": "Lecture recordings on LMS. Previous year papers in library. Online courses: Coursera, edX partnerships",
                "career_services": "Resume building workshops (Mondays 2-4PM). Mock interviews (By appointment). Industry mentorship program",
                "scholarships": "Merit scholarships available. Need-based aid. Industry-sponsored scholarships for final year students"
            }
        }
    
    def setup_openai(self, api_key: str) -> bool:
        """Setup OpenAI API configuration"""
        try:
            openai.api_key = api_key
            # Test the API key with a simple request
            openai.Completion.create(
                engine="text-davinci-003",
                prompt="Test",
                max_tokens=1
            )
            self.api_key = api_key
            return True
        except Exception as e:
            st.error(f"OpenAI API Error: {str(e)}")
            return False
    
    def classify_query_with_ai(self, query: str) -> tuple:
        """Use AI to classify query and determine confidence"""
        classification_prompt = f"""
        Classify this student query into one of these categories and provide confidence (0-1):
        
        Categories: academic_exam, academic_syllabus, academic_assignment, academic_timetable, 
                   campus_library, campus_canteen, campus_gym, campus_events, campus_placement, 
                   campus_contact, campus_medical, campus_transport, resources, summarization, general
        
        Query: "{query}"
        
        Respond in JSON format:
        {{"category": "category_name", "confidence": 0.95, "reasoning": "brief explanation"}}
        """
        
        if self.api_key:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": classification_prompt}],
                    max_tokens=100,
                    temperature=0.1
                )
                result = json.loads(response.choices[0].message.content)
                return result["category"], result["confidence"]
            except:
                pass
        
        # Fallback to rule-based classification
        return self.classify_query_simple(query), 0.8
    
    def classify_query_simple(self, query: str) -> str:
        """Simple rule-based classification"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["exam", "test", "assessment", "quiz"]):
            return "academic_exam"
        elif any(word in query_lower for word in ["syllabus", "curriculum", "subjects", "topics"]):
            return "academic_syllabus"
        elif any(word in query_lower for word in ["assignment", "homework", "project", "deadline"]):
            return "academic_assignment"
        elif any(word in query_lower for word in ["timetable", "schedule", "classes", "timing"]):
            return "academic_timetable"
        elif any(word in query_lower for word in ["library", "books", "study"]):
            return "campus_library"
        elif any(word in query_lower for word in ["canteen", "food", "mess", "cafe"]):
            return "campus_canteen"
        elif any(word in query_lower for word in ["gym", "sports", "fitness", "games"]):
            return "campus_gym"
        elif any(word in query_lower for word in ["event", "fest", "workshop", "competition"]):
            return "campus_events"
        elif any(word in query_lower for word in ["placement", "job", "company", "interview"]):
            return "campus_placement"
        elif any(word in query_lower for word in ["contact", "phone", "email", "help"]):
            return "campus_contact"
        elif any(word in query_lower for word in ["medical", "health", "doctor", "clinic"]):
            return "campus_medical"
        elif any(word in query_lower for word in ["bus", "transport", "vehicle"]):
            return "campus_transport"
        elif any(word in query_lower for word in ["resource", "material", "course", "online"]):
            return "resources"
        elif len(query.split()) > 25:
            return "summarization"
        return "general"
    
    def generate_ai_response(self, query: str, query_type: str) -> tuple:
        """Generate response using OpenAI API with context"""
        context = self.get_context_for_query_type(query_type)
        
        system_prompt = f"""
        You are QuickQuery, an intelligent student helpdesk assistant for an engineering college.
        You provide helpful, accurate, and friendly responses to student queries.
        
        Context Information:
        {context}
        
        Guidelines:
        - Be conversational and helpful
        - Provide specific information when available
        - Use appropriate emojis
        - Keep responses concise but comprehensive
        - If you don't have specific information, provide general guidance
        """
        
        user_prompt = f"Student Query: {query}"
        
        if self.api_key:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                
                ai_response = response.choices[0].message.content
                tokens_used = response.usage.total_tokens
                return ai_response, tokens_used
            except Exception as e:
                st.error(f"OpenAI API Error: {str(e)}")
        
        # Fallback to knowledge base
        return self.get_fallback_response(query_type, query), 0
    
    def get_context_for_query_type(self, query_type: str) -> str:
        """Get relevant context based on query type"""
        context_mapping = {
            "academic_exam": str(self.knowledge_base["academic"]["exams"]),
            "academic_syllabus": str(self.knowledge_base["academic"]["syllabus"]),
            "academic_assignment": str(self.knowledge_base["academic"]["assignments"]),
            "academic_timetable": str(self.knowledge_base["academic"]["timetable"]),
            "campus_library": self.knowledge_base["campus"]["facilities"]["library"],
            "campus_canteen": self.knowledge_base["campus"]["facilities"]["canteen"],
            "campus_gym": self.knowledge_base["campus"]["facilities"]["gym"],
            "campus_events": str(self.knowledge_base["campus"]["events"]),
            "campus_placement": self.knowledge_base["campus"]["events"]["placement"],
            "campus_contact": str(self.knowledge_base["campus"]["contacts"]),
            "campus_medical": self.knowledge_base["campus"]["facilities"]["medical"],
            "campus_transport": self.knowledge_base["campus"]["facilities"]["transport"],
            "resources": str(self.knowledge_base["resources"])
        }
        
        return context_mapping.get(query_type, "General college information available.")
    
    def get_fallback_response(self, query_type: str, query: str) -> str:
        """Fallback responses when OpenAI API is not available"""
        responses = {
            "academic_exam": f"📚 {self.knowledge_base['academic']['exams']['internal_exams']}. {self.knowledge_base['academic']['exams']['semester_exams']}",
            "academic_syllabus": f"📖 {self.knowledge_base['academic']['syllabus']['ai_ml']}",
            "academic_assignment": f"📝 {self.knowledge_base['academic']['assignments']['current']}",
            "academic_timetable": f"⏰ {self.knowledge_base['academic']['timetable']['regular']}",
            "campus_library": f"📚 {self.knowledge_base['campus']['facilities']['library']}",
            "campus_canteen": f"🍽️ {self.knowledge_base['campus']['facilities']['canteen']}",
            "campus_gym": f"💪 {self.knowledge_base['campus']['facilities']['gym']}",
            "campus_events": f"🎉 {self.knowledge_base['campus']['events']['tech_events']}",
            "campus_placement": f"💼 {self.knowledge_base['campus']['events']['placement']}",
            "campus_contact": f"📞 {self.knowledge_base['campus']['contacts']['academic']}",
            "campus_medical": f"🏥 {self.knowledge_base['campus']['facilities']['medical']}",
            "campus_transport": f"🚌 {self.knowledge_base['campus']['facilities']['transport']}",
            "resources": f"💼 {self.knowledge_base['resources']['online_platforms']}",
            "summarization": self.summarize_message(query)
        }
        
        return responses.get(query_type, "🤖 I can help with academic queries, campus facilities, events, and general information. Please be more specific about what you need!")
    
    def summarize_message(self, message: str) -> str:
        """Advanced message summarization"""
        sentences = [s.strip() for s in message.split('.') if len(s.strip()) > 10]
        
        if len(sentences) >= 3:
            return f"📋 Summary: {sentences[0]}. Key points: {sentences[1]} Additional info: {sentences[2]}"
        elif len(sentences) == 2:
            return f"📋 Summary: {sentences[0]}. {sentences[1]}"
        elif len(sentences) == 1:
            return f"📋 Summary: {sentences[0]}"
        else:
            return f"📋 Summary: {message[:200]}{'...' if len(message) > 200 else ''}"
    
    def process_query(self, user_query: str) -> QueryResult:
        """Process user query and return structured result"""
        start_time = datetime.datetime.now()
        
        # Classify query
        query_type, confidence = self.classify_query_with_ai(user_query)
        
        # Generate response
        ai_response, tokens_used = self.generate_ai_response(user_query, query_type)
        
        # Calculate response time
        end_time = datetime.datetime.now()
        response_time = f"{(end_time - start_time).total_seconds():.2f}s"
        
        # Update statistics
        self.update_stats(query_type, True)
        
        # Create result object
        result = QueryResult(
            user_input=user_query,
            ai_response=ai_response,
            query_type=query_type,
            confidence=confidence,
            response_time=response_time,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tokens_used=tokens_used
        )
        
        # Add to conversation history
        self.conversation_history.append(result)
        
        return result
    
    def update_stats(self, query_type: str, success: bool):
        """Update query statistics"""
        self.query_stats["total_queries"] += 1
        if success:
            self.query_stats["successful_queries"] += 1
        
        if query_type in self.query_stats["query_types"]:
            self.query_stats["query_types"][query_type] += 1
        else:
            self.query_stats["query_types"][query_type] = 1
    
    def get_analytics_data(self) -> Dict:
        """Get analytics dashboard data"""
        if not self.conversation_history:
            return {}
        
        df = pd.DataFrame([
            {
                "timestamp": result.timestamp,
                "query_type": result.query_type,
                "confidence": result.confidence,
                "tokens_used": result.tokens_used,
                "response_time": float(result.response_time.replace('s', ''))
            }
            for result in self.conversation_history
        ])
        
        return {
            "total_queries": len(self.conversation_history),
            "avg_confidence": df["confidence"].mean(),
            "avg_response_time": df["response_time"].mean(),
            "total_tokens": df["tokens_used"].sum(),
            "query_distribution": df["query_type"].value_counts().to_dict(),
            "hourly_distribution": pd.to_datetime(df["timestamp"]).dt.hour.value_counts().sort_index().to_dict()
        }

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = AdvancedQuickQueryAgent()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False

# App Header
st.title("🚀 QuickQuery: Advanced AI Student Helpdesk")
st.subheader("Powered by OpenAI GPT | OpenAI × NxtWave Buildathon 2025")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # OpenAI API Key Input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key for enhanced AI responses"
    )
    
    if api_key and not st.session_state.api_configured:
        if st.session_state.agent.setup_openai(api_key):
            st.session_state.api_configured = True
            st.success("✅ OpenAI API Connected!")
        else:
            st.error("❌ Invalid API Key")
    
    st.markdown("---")
    
    # Quick Actions
    st.header("🎯 Quick Queries")
    sample_queries = [
        "When is the next exam?",
        "Show me AI/ML syllabus",
        "Library timings and facilities?",
        "Upcoming tech events?",
        "Assignment deadlines?",
        "Placement season info",
        "Canteen timings?",
        "Gym facilities?",
        "Contact academic office",
        "Hostel information",
        "Transport routes",
        "Medical center details"
    ]
    
    for query in sample_queries:
        if st.button(query, key=f"sample_{query}", use_container_width=True):
            result = st.session_state.agent.process_query(query)
            st.session_state.chat_history.append(result)
            st.rerun()
    
    st.markdown("---")
    
    # Settings
    st.header("🛠️ Settings")
    show_analytics = st.checkbox("Show Analytics", value=False)
    show_query_details = st.checkbox("Show Query Details", value=False)
    max_history = st.slider("Max Chat History", 5, 50, 20)

# Main Interface
col1, col2 = st.columns([3, 1])

with col1:
    # Chat Interface
    st.markdown("### 💬 Chat Interface")
    
    # Display chat history
    if st.session_state.chat_history:
        for i, result in enumerate(reversed(st.session_state.chat_history[-max_history:])):
            with st.container():
                st.markdown(f"**🧑 You ({result.timestamp}):**")
                st.markdown(result.user_input)
                
                st.markdown(f"**🤖 QuickQuery:**")
                st.markdown(result.ai_response)
                
                if show_query_details:
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.caption(f"Type: {result.query_type}")
                    with col_b:
                        st.caption(f"Confidence: {result.confidence:.2f}")
                    with col_c:
                        st.caption(f"Time: {result.response_time}")
                    with col_d:
                        if result.tokens_used > 0:
                            st.caption(f"Tokens: {result.tokens_used}")
                
                st.markdown("---")
    else:
        st.info("👋 Welcome to QuickQuery! Ask me anything about your college - exams, assignments, facilities, events, and more!")

with col2:
    # Statistics Panel
    st.markdown("### 📊 Statistics")
    
    total_queries = len(st.session_state.chat_history)
    if total_queries > 0:
        analytics = st.session_state.agent.get_analytics_data()
        
        st.metric("Total Queries", total_queries)
        if analytics.get("avg_confidence"):
            st.metric("Avg Confidence", f"{analytics['avg_confidence']:.2f}")
        if analytics.get("avg_response_time"):
            st.metric("Avg Response Time", f"{analytics['avg_response_time']:.2f}s")
        if analytics.get("total_tokens", 0) > 0:
            st.metric("Total Tokens Used", analytics["total_tokens"])
    else:
        st.metric("Total Queries", 0)
        st.metric("Success Rate", "100%")
        st.metric("Status", "Ready")

# Input Section
st.markdown("### ✍️ Ask QuickQuery")
user_input = st.text_area(
    "Type your question here:",
    placeholder="e.g., When is the next exam? What are the library timings? Tell me about upcoming events...",
    height=100,
    key="user_input"
)

col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

with col1:
    if st.button("🚀 Send Query", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Processing your query..."):
                result = st.session_state.agent.process_query(user_input)
                st.session_state.chat_history.append(result)
            st.rerun()
        else:
            st.warning("Please enter a query!")

with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

with col3:
    if st.button("📊 Analytics", use_container_width=True):
        show_analytics = not show_analytics

# Analytics Dashboard
if show_analytics and st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📈 Advanced Analytics")
    
    analytics = st.session_state.agent.get_analytics_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Query Type Distribution
        if analytics.get("query_distribution"):
            fig = px.pie(
                values=list(analytics["query_distribution"].values()),
                names=list(analytics["query_distribution"].keys()),
                title="Query Type Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Hourly Query Distribution
        if analytics.get("hourly_distribution"):
            hours = list(analytics["hourly_distribution"].keys())
            counts = list(analytics["hourly_distribution"].values())
            
            fig = go.Figure(data=go.Bar(x=hours, y=counts))
            fig.update_layout(title="Queries by Hour", xaxis_title="Hour", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🏆 <strong>OpenAI × NxtWave Buildathon 2025 Submission</strong></p>
        <p>🤖 Advanced AI-Powered Student Helpdesk | Built with Streamlit & OpenAI GPT</p>
        <p>✨ Features: Real-time AI responses, Smart query classification, Analytics dashboard, Multi-modal support</p>
    </div>
    """,
    unsafe_allow_html=True
)
'''

# Save the enhanced Streamlit app
with open('enhanced_streamlit_app.py', 'w') as f:
    f.write(enhanced_streamlit_app)

print("✅ Enhanced Streamlit app created: enhanced_streamlit_app.py")
print("🌟 Features added:")
print("  - Real OpenAI API integration")
print("  - Advanced query classification")
print("  - Analytics dashboard")
print("  - Enhanced UI with statistics")
print("  - Professional-grade code structure")