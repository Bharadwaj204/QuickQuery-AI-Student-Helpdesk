import openai
import datetime
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import asyncio

# Fix for OpenAI API v1 migration
# Removed due to ModuleNotFoundError for openai.api_resources

# You can downgrade openai package to 0.28.0 to avoid migration issues:
# pip install openai==0.28.0

@dataclass
class QueryAnalytics:
    """Data class for query analytics"""
    query_id: str
    user_input: str
    ai_response: str
    query_type: str
    confidence: float
    response_time_ms: int
    tokens_used: int
    timestamp: str
    user_satisfaction: Optional[int] = None

class AdvancedQuickQueryAgent:
    """
    Production-ready AI Student Helpdesk Agent with OpenAI GPT Integration

    Features:
    - Real OpenAI API integration
    - Advanced query classification
    - Context-aware responses
    - Analytics and logging
    - Error handling and fallbacks
    - Async support for scalability
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.conversation_history: List[QueryAnalytics] = []
        self.session_stats = {
            "queries_processed": 0,
            "successful_queries": 0,
            "total_tokens_used": 0,
            "average_response_time": 0,
            "query_types": {},
            "session_start": datetime.datetime.now()
        }

        # Enhanced knowledge base with more comprehensive information
        self.knowledge_base = {
            "academic": {
                "exams": {
                    "schedule": {
                        "internal_1": "Internal Exam 1: September 20-25, 2025",
                        "internal_2": "Internal Exam 2: November 15-20, 2025", 
                        "semester": "End Semester Exams: December 10-20, 2025",
                        "practical": "Practical Exams: December 5-9, 2025"
                    },
                    "rules": [
                        "ID card mandatory for all exams",
                        "Arrive 30 minutes before exam time",
                        "No mobile phones or electronic devices allowed",
                        "Scientific calculator allowed for specific subjects only",
                        "Blue/black pen mandatory, pencil only for diagrams"
                    ],
                    "entrance_exams": {
                        "gate_2026": "February 1-16, 2026",
                        "jee_main": "January 2026",
                        "cat_2025": "November 26, 2025",
                        "neet_2026": "May 5, 2026"
                    }
                },
                "syllabus": {
                    "ai_ml": {
                        "semester_7": [
                            "Neural Networks and Deep Learning (30%)",
                            "Natural Language Processing (25%)",
                            "Computer Vision (20%)",
                            "Machine Learning Operations (MLOps) (15%)",
                            "AI Ethics and Fairness (10%)"
                        ],
                        "practical_components": [
                            "TensorFlow/PyTorch projects",
                            "Real-world dataset analysis",
                            "Model deployment exercises",
                            "Research paper implementation"
                        ]
                    },
                    "cs_core": {
                        "data_structures": "Arrays, Linked Lists, Trees, Graphs, Hashing",
                        "algorithms": "Sorting, Searching, Dynamic Programming, Greedy Algorithms",
                        "database": "SQL, NoSQL, Database Design, Transactions, Normalization",
                        "operating_systems": "Process Management, Memory Management, File Systems",
                        "computer_networks": "TCP/IP, OSI Model, Routing, Network Security"
                    }
                },
                "assignments": {
                    "current_deadlines": [
                        {"name": "AI Project Phase 1", "due": "September 30, 2025", "weightage": "25%"},
                        {"name": "ML Assignment 2", "due": "October 5, 2025", "weightage": "15%"},
                        {"name": "Database Project", "due": "October 15, 2025", "weightage": "20%"},
                        {"name": "Network Security Assignment", "due": "October 22, 2025", "weightage": "15%"}
                    ],
                    "submission_guidelines": [
                        "Submit only through college LMS portal",
                        "Late submissions: -10% marks per day",
                        "Plagiarism results in zero marks",
                        "Group projects require individual contribution reports",
                        "Code submissions must include documentation"
                    ]
                },
                "resources": {
                    "online_platforms": [
                        "College LMS: portal.college.edu",
                        "GitHub Student Pack (free for students)",
                        "Microsoft Office 365 for Education",
                        "Coursera for Campus (free courses)",
                        "IEEE Xplore Digital Library access"
                    ],
                    "study_materials": [
                        "Recorded lecture videos on LMS",
                        "Previous year question papers (library)",
                        "Reference books available in digital format",
                        "Research paper access through college subscriptions"
                    ]
                }
            },
            "campus": {
                "facilities": {
                    "library": {
                        "timings": "6:00 AM - 10:00 PM (Mon-Sat), 8:00 AM - 8:00 PM (Sun)",
                        "capacity": "500 students",
                        "collections": "50,000+ books, 200+ journals, 10,000+ e-books",
                        "services": ["WiFi", "Printing", "Scanning", "Group study rooms", "Silent zones"],
                        "digital_resources": ["IEEE Xplore", "ACM Digital Library", "Springer", "Elsevier"]
                    },
                    "dining": {
                        "main_canteen": {
                            "timings": "8:00 AM - 8:00 PM",
                            "cuisine": "South Indian, North Indian, Chinese",
                            "capacity": "300 seats",
                            "special_features": ["Vegan options", "Jain food", "Diet meals"]
                        },
                        "coffee_shop": {
                            "timings": "7:00 AM - 9:00 PM", 
                            "offerings": ["Coffee", "Tea", "Snacks", "Sandwiches", "Pastries"]
                        },
                        "night_canteen": {
                            "timings": "8:00 PM - 11:00 PM",
                            "offerings": ["Light snacks", "Beverages", "Instant noodles"]
                        }
                    },
                    "sports_fitness": {
                        "gym": {
                            "timings": "6:00 AM - 10:00 PM",
                            "equipment": ["Cardio machines", "Weight training", "Functional fitness"],
                            "facilities": ["Locker rooms", "Shower facilities", "First aid"],
                            "membership": "Free for all students"
                        },
                        "sports_facilities": [
                            "Basketball court (2 courts)",
                            "Badminton court (4 courts)", 
                            "Tennis court (2 courts)",
                            "Cricket ground (1 full size)",
                            "Football ground (1 full size)",
                            "Swimming pool (50m, seasonal)"
                        ]
                    },
                    "accommodation": {
                        "boys_hostel": {
                            "capacity": "500 students",
                            "room_types": ["Single occupancy", "Twin sharing"],
                            "facilities": ["WiFi", "Laundry", "Common room", "Study hall"],
                            "mess_timings": "Breakfast: 7-9 AM, Lunch: 12-2 PM, Dinner: 7-9 PM"
                        },
                        "girls_hostel": {
                            "capacity": "300 students", 
                            "room_types": ["Single occupancy", "Twin sharing"],
                            "facilities": ["WiFi", "Laundry", "Common room", "Study hall"],
                            "security": "24/7 security, CCTV, Biometric access"
                        }
                    }
                },
                "transportation": {
                    "college_buses": {
                        "routes": 12,
                        "timings": "7:00 AM - 7:00 PM",
                        "frequency": "Every 30 minutes during peak hours",
                        "monthly_pass": "₹500 (Regular), ₹800 (AC buses)",
                        "coverage": "Major areas across the city"
                    },
                    "parking": {
                        "two_wheeler": "Free for students with valid stickers",
                        "four_wheeler": "₹20 per day for visitors",
                        "bicycle": "Free bicycle stands available"
                    }
                },
                "events": {
                    "upcoming": [
                        {
                            "name": "TechFest 2025",
                            "dates": "October 15-17, 2025",
                            "events": ["Hackathon", "Tech talks", "Project exhibitions", "Coding competitions"]
                        },
                        {
                            "name": "AI Workshop Series",
                            "date": "September 25, 2025",
                            "topics": ["Introduction to GPT models", "Computer vision applications", "ML in industry"]
                        },
                        {
                            "name": "Coding Championship",
                            "date": "October 1, 2025", 
                            "prizes": "₹50,000 total prize pool",
                            "registration": "Free for college students"
                        }
                    ],
                    "placement_season": {
                        "timeline": "November 2025 - March 2026",
                        "companies": ["Google", "Microsoft", "Amazon", "Adobe", "Infosys", "TCS", "Wipro"],
                        "preparation": [
                            "Resume building workshops (every Monday 2-4 PM)",
                            "Mock interviews (by appointment)",
                            "Aptitude test preparation",
                            "Technical interview preparation"
                        ]
                    }
                }
            },
            "contacts": {
                "emergency": {
                    "security": "ext-911 / security@college.edu",
                    "medical": "ext-100 / medical@college.edu", 
                    "fire_safety": "ext-101 / fire@college.edu",
                    "admin": "ext-500 / admin@college.edu"
                },
                "academic": {
                    "dean": "dean@college.edu / ext-001",
                    "hod_cse": "hod.cse@college.edu / ext-201",
                    "academic_office": "academic@college.edu / ext-101",
                    "examination_cell": "exams@college.edu / ext-102"
                },
                "student_services": {
                    "hostel_office": "hostel@college.edu / ext-301",
                    "placement_cell": "placement@college.edu / ext-401",
                    "student_counselor": "counselor@college.edu / ext-601",
                    "it_helpdesk": "ithelpdesk@college.edu / ext-201"
                }
            }
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

        if api_key:
            self.setup_openai(api_key)

    def setup_openai(self, api_key: str) -> bool:
        """Initialize OpenAI API configuration"""
        try:
            openai.api_key = api_key
            self.api_key = api_key

            # Test API connection
            test_response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )

            self.logger.info("OpenAI API connection successful")
            return True

        except Exception as e:
            self.logger.error(f"OpenAI API setup failed: {str(e)}")
            return False

    def classify_query_advanced(self, query: str) -> Tuple[str, float]:
        """
        Advanced query classification using OpenAI API with confidence scoring
        """
        classification_prompt = f"""
        As an expert classifier for a student helpdesk system, classify this query into one of these categories.
        Also provide a confidence score (0.0 to 1.0) and brief reasoning.

        Categories:
        - academic_exam: Questions about exam schedules, rules, entrance exams
        - academic_syllabus: Questions about course content, curriculum, subjects
        - academic_assignment: Questions about assignments, projects, deadlines  
        - academic_timetable: Questions about class schedules, lab timings
        - campus_library: Questions about library services, timings, resources
        - campus_dining: Questions about canteen, food, mess facilities
        - campus_sports: Questions about gym, sports facilities, games
        - campus_hostel: Questions about accommodation, hostel facilities
        - campus_transport: Questions about buses, parking, transportation
        - campus_events: Questions about events, festivals, competitions
        - campus_placement: Questions about jobs, placements, career services
        - campus_contact: Questions about contact information, phone numbers
        - resources: Questions about online platforms, study materials
        - summarization: Long text that needs to be summarized
        - general: General queries that don't fit other categories

        Query: "{query}"

        Respond in JSON format:
        {{
            "category": "category_name",
            "confidence": 0.95,
            "reasoning": "Brief explanation for this classification"
        }}
        """

        if self.api_key:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": classification_prompt}],
                    max_tokens=150,
                    temperature=0.1
                )

                result = json.loads(response.choices[0].message.content)
                return result["category"], result["confidence"]

            except Exception as e:
                self.logger.error(f"AI classification failed: {str(e)}")

        # Fallback to rule-based classification
        return self._classify_rule_based(query), 0.75

    def _classify_rule_based(self, query: str) -> str:
        """Fallback rule-based classification"""
        query_lower = query.lower()

        # Academic queries
        if any(word in query_lower for word in ["exam", "test", "assessment", "quiz", "gate", "jee"]):
            return "academic_exam"
        elif any(word in query_lower for word in ["syllabus", "curriculum", "subjects", "topics", "course"]):
            return "academic_syllabus"
        elif any(word in query_lower for word in ["assignment", "homework", "project", "deadline", "submission"]):
            return "academic_assignment"
        elif any(word in query_lower for word in ["timetable", "schedule", "classes", "timing", "class"]):
            return "academic_timetable"

        # Campus facilities
        elif any(word in query_lower for word in ["library", "books", "study", "reading"]):
            return "campus_library"
        elif any(word in query_lower for word in ["canteen", "food", "mess", "cafe", "dining", "restaurant"]):
            return "campus_dining"
        elif any(word in query_lower for word in ["gym", "sports", "fitness", "games", "basketball", "badminton"]):
            return "campus_sports"
        elif any(word in query_lower for word in ["hostel", "accommodation", "room", "residence"]):
            return "campus_hostel"
        elif any(word in query_lower for word in ["bus", "transport", "vehicle", "parking"]):
            return "campus_transport"

        # Events and services
        elif any(word in query_lower for word in ["event", "fest", "workshop", "competition", "seminar"]):
            return "campus_events"
        elif any(word in query_lower for word in ["placement", "job", "company", "interview", "career"]):
            return "campus_placement"
        elif any(word in query_lower for word in ["contact", "phone", "email", "help", "support"]):
            return "campus_contact"
        elif any(word in query_lower for word in ["resource", "material", "online", "platform", "lms"]):
            return "resources"

        # Special cases
        elif len(query.split()) > 25:
            return "summarization"

        return "general"

    def generate_ai_response(self, query: str, query_type: str, conversation_context: List[str] = None) -> Tuple[str, int]:
        """
        Generate contextual AI response using OpenAI GPT with conversation history
        """
        # Get relevant context from knowledge base
        context_info = self._get_contextual_information(query_type)

        # Build conversation context
        context_messages = []
        if conversation_context:
            for ctx in conversation_context[-3:]:  # Last 3 exchanges for context
                context_messages.append(ctx)

        # Create system prompt
        system_prompt = f"""
        You are QuickQuery, an intelligent and helpful AI assistant for engineering college students.
        You specialize in providing accurate, friendly, and comprehensive information about academic and campus life.

        Current Context: {query_type}
        Relevant Information: {context_info}

        Guidelines:
        - Provide specific, actionable information when available
        - Be conversational and empathetic to student needs  
        - Use appropriate emojis to make responses engaging
        - If you don't have specific information, provide general guidance
        - Keep responses comprehensive but concise (under 300 words)
        - Always prioritize accuracy over speculation
        - Format information clearly with bullet points when appropriate
        """

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation context
        if context_messages:
            messages.extend(context_messages)

        # Add current query
        messages.append({"role": "user", "content": query})

        if self.api_key:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=400,
                    temperature=0.7,
                    presence_penalty=0.1,
                    frequency_penalty=0.1
                )

                ai_response = response.choices[0].message.content
                tokens_used = response.usage.total_tokens

                return ai_response, tokens_used

            except Exception as e:
                self.logger.error(f"OpenAI response generation failed: {str(e)}")

        # Fallback response
        return self._get_fallback_response(query_type, query), 0

    def _get_contextual_information(self, query_type: str) -> str:
        """Extract relevant information from knowledge base based on query type"""

        context_mapping = {
            "academic_exam": json.dumps(self.knowledge_base["academic"]["exams"], indent=2),
            "academic_syllabus": json.dumps(self.knowledge_base["academic"]["syllabus"], indent=2),
            "academic_assignment": json.dumps(self.knowledge_base["academic"]["assignments"], indent=2),
            "campus_library": json.dumps(self.knowledge_base["campus"]["facilities"]["library"], indent=2),
            "campus_dining": json.dumps(self.knowledge_base["campus"]["facilities"]["dining"], indent=2),
            "campus_sports": json.dumps(self.knowledge_base["campus"]["facilities"]["sports_fitness"], indent=2),
            "campus_hostel": json.dumps(self.knowledge_base["campus"]["facilities"]["accommodation"], indent=2),
            "campus_transport": json.dumps(self.knowledge_base["campus"]["transportation"], indent=2),
            "campus_events": json.dumps(self.knowledge_base["campus"]["events"], indent=2),
            "campus_contact": json.dumps(self.knowledge_base["contacts"], indent=2),
            "resources": json.dumps(self.knowledge_base["academic"]["resources"], indent=2)
        }

        return context_mapping.get(query_type, "General college information available.")

    def _get_fallback_response(self, query_type: str, query: str) -> str:
        """Generate fallback response when OpenAI API is unavailable"""

        fallback_responses = {
            "academic_exam": f"📚 **Exam Information:**\n{self.knowledge_base['academic']['exams']['schedule']['semester']}\n{self.knowledge_base['academic']['exams']['schedule']['internal_1']}",

            "academic_syllabus": f"📖 **AI/ML Syllabus:** {', '.join(self.knowledge_base['academic']['syllabus']['ai_ml']['semester_7'])}",

            "academic_assignment": f"📝 **Current Assignments:**\n" + "\n".join([f"• {item['name']} - Due: {item['due']}" for item in self.knowledge_base['academic']['assignments']['current_deadlines'][:3]]),

            "campus_library": f"📚 **Library Information:**\nTimings: {self.knowledge_base['campus']['facilities']['library']['timings']}\nCapacity: {self.knowledge_base['campus']['facilities']['library']['capacity']}",

            "campus_dining": f"🍽️ **Dining Facilities:**\nMain Canteen: {self.knowledge_base['campus']['facilities']['dining']['main_canteen']['timings']}\nCoffee Shop: {self.knowledge_base['campus']['facilities']['dining']['coffee_shop']['timings']}",

            "campus_events": "🎉 **Upcoming Events:** TechFest 2025 (Oct 15-17), AI Workshop (Sept 25), Coding Competition (Oct 1)",

            "campus_contact": f"📞 **Important Contacts:**\nAcademic Office: {self.knowledge_base['contacts']['academic']['academic_office']}\nIT Helpdesk: {self.knowledge_base['contacts']['student_services']['it_helpdesk']}",

            "summarization": self._summarize_text(query)
        }

        return fallback_responses.get(query_type, "🤖 I can help with academic queries, campus facilities, events, and general information. Please be more specific about what you need!")

    def _summarize_text(self, text: str) -> str:
        """Simple text summarization for fallback"""
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]

        if len(sentences) >= 3:
            key_points = sentences[:2]
            return f"📋 **Summary:** {'. '.join(key_points)}. [Total: {len(sentences)} sentences]"
        elif len(sentences) >= 2:
            return f"📋 **Summary:** {sentences[0]}. {sentences[1]}"
        else:
            return f"📋 **Summary:** {text[:150]}{'...' if len(text) > 150 else ''}"

    def process_query(self, user_query: str, conversation_context: List[str] = None) -> QueryAnalytics:
        """
        Main method to process user queries with full analytics
        """
        start_time = datetime.datetime.now()
        query_id = f"q_{int(start_time.timestamp())}"

        try:
            # Classify the query
            query_type, confidence = self.classify_query_advanced(user_query)

            # Generate AI response
            ai_response, tokens_used = self.generate_ai_response(user_query, query_type, conversation_context)

            # Calculate response time
            end_time = datetime.datetime.now()
            response_time_ms = int((end_time - start_time).total_seconds() * 1000)

            # Create analytics object
            analytics = QueryAnalytics(
                query_id=query_id,
                user_input=user_query,
                ai_response=ai_response,
                query_type=query_type,
                confidence=confidence,
                response_time_ms=response_time_ms,
                tokens_used=tokens_used,
                timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S")
            )

            # Update session statistics
            self._update_session_stats(query_type, response_time_ms, tokens_used, True)

            # Add to conversation history
            self.conversation_history.append(analytics)

            return analytics

        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")

            # Return error response
            return QueryAnalytics(
                query_id=query_id,
                user_input=user_query,
                ai_response=f"❌ I encountered an error processing your query. Please try again or rephrase your question.",
                query_type="error",
                confidence=0.0,
                response_time_ms=0,
                tokens_used=0,
                timestamp=start_time.strftime("%Y-%m-%d %H:%M:%S")
            )

    def _update_session_stats(self, query_type: str, response_time_ms: int, tokens_used: int, success: bool):
        """Update session statistics"""
        self.session_stats["queries_processed"] += 1

        if success:
            self.session_stats["successful_queries"] += 1

        self.session_stats["total_tokens_used"] += tokens_used

        # Update average response time
        total_time = (self.session_stats["average_response_time"] * (self.session_stats["queries_processed"] - 1) + response_time_ms)
        self.session_stats["average_response_time"] = total_time / self.session_stats["queries_processed"]

        # Update query type distribution
        if query_type in self.session_stats["query_types"]:
            self.session_stats["query_types"][query_type] += 1
        else:
            self.session_stats["query_types"][query_type] = 1

    def get_analytics_summary(self) -> Dict:
        """Get comprehensive analytics summary"""
        if not self.conversation_history:
            return {
                "total_queries": 0,
                "success_rate": 100.0,
                "average_confidence": 0.0,
                "average_response_time": 0,
                "total_tokens": 0
            }

        total_queries = len(self.conversation_history)
        successful_queries = len([q for q in self.conversation_history if q.query_type != "error"])

        return {
            "total_queries": total_queries,
            "success_rate": (successful_queries / total_queries) * 100,
            "average_confidence": sum(q.confidence for q in self.conversation_history) / total_queries,
            "average_response_time": sum(q.response_time_ms for q in self.conversation_history) / total_queries,
            "total_tokens": sum(q.tokens_used for q in self.conversation_history),
            "query_distribution": self.session_stats["query_types"],
            "session_duration": (datetime.datetime.now() - self.session_stats["session_start"]).total_seconds()
        }

    def export_conversation_data(self) -> List[Dict]:
        """Export conversation history for analysis"""
        return [asdict(query) for query in self.conversation_history]


# Utility functions for the application
def format_response_for_display(response: str) -> str:
    """Format AI response for better display in Streamlit"""
    # Add proper line breaks and formatting
    formatted = response.replace("\n", "\n")
    formatted = formatted.replace("**", "**")
    return formatted

def calculate_usage_cost(tokens_used: int, model: str = "gpt-3.5-turbo") -> float:
    """Calculate approximate API usage cost"""
    # Pricing as of 2025 (approximate)
    pricing = {
        "gpt-3.5-turbo": 0.002 / 1000,  # $0.002 per 1K tokens
        "gpt-4": 0.03 / 1000,  # $0.03 per 1K tokens
    }

    return tokens_used * pricing.get(model, 0.002 / 1000)

if __name__ == "__main__":
    # Demo usage
    agent = AdvancedQuickQueryAgent()

    # Test queries
    test_queries = [
        "When is the next exam?",
        "Tell me about the AI/ML syllabus",
        "What are the library timings?",
        "How do I contact the placement office?"
    ]

    print("🚀 QuickQuery Advanced Agent Demo")
    print("=" * 50)

    for query in test_queries:
        print(f"\nUser: {query}")
        result = agent.process_query(query)
        print(f"Assistant: {result.ai_response}")
        print(f"Type: {result.query_type} | Confidence: {result.confidence:.2f} | Time: {result.response_time_ms}ms")

    # Print analytics
    analytics = agent.get_analytics_summary()
    print(f"\n📊 Session Analytics:")
    print(f"Total Queries: {analytics['total_queries']}")
    print(f"Success Rate: {analytics['success_rate']:.1f}%")
    print(f"Average Response Time: {analytics['average_response_time']:.0f}ms")
