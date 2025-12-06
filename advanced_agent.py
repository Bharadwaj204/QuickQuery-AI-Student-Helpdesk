"""
Advanced QuickQuery AI Agent with Modern OpenAI Integration
Production-ready implementation with streaming, analytics, and security
"""

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
import asyncio
import datetime
import json
import logging
import hashlib
import time
from typing import Dict, List, Optional, AsyncGenerator, Union, Any
from dataclasses import dataclass, asdict
from functools import lru_cache
import re


@dataclass
class QueryAnalytics:
    """Enhanced data class for query analytics with validation"""
    query_id: str
    user_input: str
    ai_response: str
    query_type: str
    confidence: float
    response_time_ms: int
    tokens_used: int
    timestamp: str
    user_satisfaction: Optional[int] = None
    cost_estimate: Optional[float] = None
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.response_time_ms < 0:
            raise ValueError("Response time cannot be negative")


class SecurityValidator:
    """Security validation utilities"""
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        # Remove potential harmful characters
        sanitized = re.sub(r'[<>"\']', '', text)
        
        # Limit length to prevent DoS
        if len(sanitized) > 2000:
            sanitized = sanitized[:2000]
            
        return sanitized.strip()
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate OpenAI API key format"""
        if not api_key:
            return False
        
        # Basic OpenAI API key format validation
        pattern = r'^sk-[a-zA-Z0-9]{48,}$'
        return bool(re.match(pattern, api_key))


class KnowledgeBase:
    """Structured knowledge base with caching and search capabilities"""
    
    def __init__(self):
        self.data = self._load_knowledge_data()
        self._search_cache = {}
        
    @lru_cache(maxsize=100)
    def get_context_for_type(self, query_type: str) -> str:
        """Get relevant context based on query type with caching"""
        context_mapping = {
            "academic_exam": json.dumps(self.data["academic"]["exams"], indent=2),
            "academic_syllabus": json.dumps(self.data["academic"]["syllabus"], indent=2),
            "academic_assignment": json.dumps(self.data["academic"]["assignments"], indent=2),
            "campus_library": json.dumps(self.data["campus"]["facilities"]["library"], indent=2),
            "campus_dining": json.dumps(self.data["campus"]["facilities"]["dining"], indent=2),
            "campus_sports": json.dumps(self.data["campus"]["facilities"]["sports_fitness"], indent=2),
            "campus_hostel": json.dumps(self.data["campus"]["facilities"]["accommodation"], indent=2),
            "campus_transport": json.dumps(self.data["campus"]["transportation"], indent=2),
            "campus_events": json.dumps(self.data["campus"]["events"], indent=2),
            "campus_contact": json.dumps(self.data["contacts"], indent=2),
            "resources": json.dumps(self.data["academic"]["resources"], indent=2)
        }
        
        return context_mapping.get(query_type, "General college information available.")
    
    def _load_knowledge_data(self) -> Dict[str, Any]:
        """Load comprehensive knowledge base"""
        return {
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
                        "Scientific calculator allowed for specific subjects only"
                    ],
                    "entrance_exams": {
                        "gate_2026": "February 1-16, 2026",
                        "jee_main": "January 2026",
                        "cat_2025": "November 26, 2025"
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
                        ]
                    },
                    "cs_core": {
                        "data_structures": "Arrays, Linked Lists, Trees, Graphs, Hashing",
                        "algorithms": "Sorting, Searching, Dynamic Programming, Greedy Algorithms",
                        "database": "SQL, NoSQL, Database Design, Transactions, Normalization"
                    }
                },
                "assignments": {
                    "current_deadlines": [
                        {"name": "AI Project Phase 1", "due": "September 30, 2025", "weightage": "25%"},
                        {"name": "ML Assignment 2", "due": "October 5, 2025", "weightage": "15%"},
                        {"name": "Database Project", "due": "October 15, 2025", "weightage": "20%"}
                    ]
                },
                "resources": {
                    "online_platforms": [
                        "College LMS: portal.college.edu",
                        "GitHub Student Pack (free for students)",
                        "Microsoft Office 365 for Education"
                    ]
                }
            },
            "campus": {
                "facilities": {
                    "library": {
                        "timings": "6:00 AM - 10:00 PM (Mon-Sat), 8:00 AM - 8:00 PM (Sun)",
                        "capacity": "500 students",
                        "collections": "50,000+ books, 200+ journals, 10,000+ e-books",
                        "services": ["WiFi", "Printing", "Scanning", "Group study rooms"]
                    },
                    "dining": {
                        "main_canteen": {
                            "timings": "8:00 AM - 8:00 PM",
                            "cuisine": "South Indian, North Indian, Chinese",
                            "capacity": "300 seats"
                        },
                        "coffee_shop": {
                            "timings": "7:00 AM - 9:00 PM",
                            "offerings": ["Coffee", "Tea", "Snacks", "Sandwiches"]
                        }
                    },
                    "sports_fitness": {
                        "gym": {
                            "timings": "6:00 AM - 10:00 PM",
                            "equipment": ["Cardio machines", "Weight training"],
                            "membership": "Free for all students"
                        }
                    },
                    "accommodation": {
                        "boys_hostel": {
                            "capacity": "500 students",
                            "facilities": ["WiFi", "Laundry", "Common room"]
                        },
                        "girls_hostel": {
                            "capacity": "300 students",
                            "facilities": ["WiFi", "Laundry", "Common room"]
                        }
                    }
                },
                "transportation": {
                    "college_buses": {
                        "routes": 12,
                        "timings": "7:00 AM - 7:00 PM",
                        "monthly_pass": "₹500 (Regular), ₹800 (AC buses)"
                    }
                },
                "events": {
                    "upcoming": [
                        {
                            "name": "TechFest 2025",
                            "dates": "October 15-17, 2025",
                            "events": ["Hackathon", "Tech talks", "Project exhibitions"]
                        },
                        {
                            "name": "AI Workshop Series",
                            "date": "September 25, 2025",
                            "topics": ["Introduction to GPT models", "Computer vision applications"]
                        }
                    ],
                    "placement_season": {
                        "timeline": "November 2025 - March 2026",
                        "companies": ["Google", "Microsoft", "Amazon", "Adobe", "Infosys"]
                    }
                }
            },
            "contacts": {
                "academic": {
                    "dean": "dean@college.edu / ext-001",
                    "hod_cse": "hod.cse@college.edu / ext-201",
                    "academic_office": "academic@college.edu / ext-101"
                },
                "student_services": {
                    "hostel_office": "hostel@college.edu / ext-301",
                    "placement_cell": "placement@college.edu / ext-401",
                    "it_helpdesk": "ithelpdesk@college.edu / ext-201"
                },
                "emergency": {
                    "security": "ext-911 / security@college.edu",
                    "medical": "ext-100 / medical@college.edu"
                }
            }
        }


class EnhancedQuickQueryAgent:
    """
    Production-ready AI Student Helpdesk Agent
    
    Features:
    - Modern OpenAI API v1+ integration
    - Real-time streaming responses
    - Advanced query classification
    - Comprehensive security and validation
    - Performance monitoring and analytics
    - Intelligent caching and rate limiting
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.api_key: Optional[str] = None
        self.client: Optional[AsyncOpenAI] = None
        self.sync_client: Optional[OpenAI] = None
        
        # Initialize components
        self.knowledge_base = KnowledgeBase()
        self.security = SecurityValidator()
        self.conversation_history: List[QueryAnalytics] = []
        
        # Session statistics
        self.session_stats = {
            "queries_processed": 0,
            "successful_queries": 0,
            "total_tokens_used": 0,
            "average_response_time": 0,
            "query_types": {},
            "session_start": datetime.datetime.now(),
            "api_calls_made": 0,
            "cache_hits": 0
        }
        
        # Performance settings
        self.max_tokens = 500
        self.temperature = 0.7
        self.request_timeout = 30
        
        # Setup logging
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger(f"{__name__}.{id(self)}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def setup_openai(self, api_key: str) -> bool:
        """Initialize OpenAI API with proper validation"""
        try:
            # Validate API key format
            if not self.security.validate_api_key(api_key):
                self.logger.error("Invalid API key format")
                return False
            
            # Initialize clients
            self.api_key = api_key
            self.sync_client = OpenAI(api_key=api_key, timeout=self.request_timeout)
            self.client = AsyncOpenAI(api_key=api_key, timeout=self.request_timeout)
            
            # Test connection
            test_response = self.sync_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            
            self.logger.info("OpenAI API connection successful")
            return True
            
        except Exception as e:
            self.logger.error(f"OpenAI API setup failed: {str(e)}")
            self.api_key = None
            self.client = None
            self.sync_client = None
            return False
    
    @lru_cache(maxsize=50)
    def classify_query(self, query: str) -> tuple[str, float]:
        """Classify query with caching for performance"""
        try:
            # Sanitize input
            clean_query = self.security.sanitize_input(query)
            
            # Rule-based classification with confidence scoring
            query_lower = clean_query.lower()
            
            # Define classification rules with confidence scores
            classification_rules = [
                (["exam", "test", "assessment", "quiz"], "academic_exam", 0.9),
                (["syllabus", "curriculum", "course"], "academic_syllabus", 0.85),
                (["assignment", "homework", "project", "deadline"], "academic_assignment", 0.88),
                (["timetable", "schedule", "classes"], "academic_timetable", 0.82),
                (["library", "books", "study"], "campus_library", 0.8),
                (["canteen", "food", "mess", "cafe"], "campus_dining", 0.8),
                (["gym", "sports", "fitness"], "campus_sports", 0.8),
                (["hostel", "accommodation", "room"], "campus_hostel", 0.85),
                (["bus", "transport", "vehicle"], "campus_transport", 0.8),
                (["event", "fest", "workshop"], "campus_events", 0.8),
                (["placement", "job", "company"], "campus_placement", 0.85),
                (["contact", "phone", "email"], "campus_contact", 0.75),
                (["resource", "material", "online"], "resources", 0.7)
            ]
            
            # Find best match
            for keywords, category, confidence in classification_rules:
                if any(keyword in query_lower for keyword in keywords):
                    return category, confidence
            
            # Special case for long text
            if len(clean_query.split()) > 25:
                return "summarization", 0.6
            
            return "general", 0.5
            
        except Exception as e:
            self.logger.error(f"Query classification failed: {str(e)}")
            return "general", 0.3
    
    async def generate_response_stream(self, query: str, query_type: str, 
                                     conversation_context: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[str, None]:
        """Generate streaming AI response with proper error handling"""
        try:
            # Get relevant context
            context_info = self.knowledge_base.get_context_for_type(query_type)
            
            # Build messages for OpenAI with proper typing
            messages: List[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": f"""You are QuickQuery, an intelligent AI assistant for engineering college students.
                    
Current Context: {query_type}
Relevant Information: {context_info}

Guidelines:
- Provide specific, actionable information
- Be conversational and helpful
- Use appropriate emojis
- Keep responses under 300 words
- Format information clearly with bullet points when appropriate"""
                }
            ]
            
            # Add conversation context with proper typing
            if conversation_context:
                for ctx in conversation_context[-6:]:  # Last 3 exchanges
                    if ctx.get("role") in ["user", "assistant"] and ctx.get("content"):
                        messages.append({
                            "role": ctx["role"],  # type: ignore
                            "content": ctx["content"]
                        })
            
            # Add current query
            messages.append({"role": "user", "content": query})
            
            if self.client:
                # Make API call with streaming
                stream = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=True
                )
                
                self.session_stats["api_calls_made"] += 1
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                # Fallback response
                fallback = self._get_fallback_response(query_type, query)
                yield fallback
                
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            error_msg = "❌ I encountered an error processing your query. Please try again or rephrase your question."
            yield error_msg
    
    def _get_fallback_response(self, query_type: str, query: str) -> str:
        """Generate fallback response when API is unavailable"""
        fallback_responses = {
            "academic_exam": "📚 **Exam Information:** Internal Exams: Sept 20-25, Nov 15-20, 2025. Semester Exams: Dec 10-20, 2025.",
            "academic_syllabus": "📖 **AI/ML Syllabus:** Neural Networks (30%), NLP (25%), Computer Vision (20%), MLOps (15%), AI Ethics (10%)",
            "academic_assignment": "📝 **Current Assignments:** AI Project (Sept 30), ML Assignment 2 (Oct 5), Database Project (Oct 15)",
            "campus_library": "📚 **Library:** Open 6AM-10PM (Mon-Sat), 8AM-8PM (Sun). 500 student capacity with WiFi and study rooms.",
            "campus_dining": "🍽️ **Dining:** Main Canteen 8AM-8PM, Coffee Shop 7AM-9PM. Multiple cuisine options available.",
            "campus_events": "🎉 **Events:** TechFest 2025 (Oct 15-17), AI Workshop (Sept 25), Coding Competition (Oct 1)",
            "campus_contact": "📞 **Contacts:** Academic Office: ext-101, IT Helpdesk: ext-201, Placement: ext-401",
            "summarization": self._summarize_text(query)
        }
        
        return fallback_responses.get(query_type, 
            "🤖 I can help with academic queries, campus facilities, events, and general information. Please be more specific!")
    
    def _summarize_text(self, text: str) -> str:
        """Simple text summarization for fallback"""
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        
        if len(sentences) >= 3:
            return f"📋 **Summary:** {sentences[0]}. Key points: {sentences[1]}."
        elif len(sentences) >= 2:
            return f"📋 **Summary:** {sentences[0]}. {sentences[1]}"
        else:
            return f"📋 **Summary:** {text[:150]}{'...' if len(text) > 150 else ''}"
    
    async def process_query_stream(self, user_query: str, 
                                 conversation_context: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[Union[str, QueryAnalytics], None]:
        """Process query with streaming response and analytics"""
        start_time = time.perf_counter()
        timestamp = datetime.datetime.now()
        query_id = f"q_{int(timestamp.timestamp())}_{id(self)}"
        
        try:
            # Sanitize and validate input
            clean_query = self.security.sanitize_input(user_query)
            if not clean_query:
                yield "❌ Please provide a valid question."
                return
            
            # Classify query
            query_type, confidence = self.classify_query(clean_query)
            
            # Stream response
            full_response = ""
            async for chunk in self.generate_response_stream(clean_query, query_type, conversation_context):
                full_response += chunk
                yield chunk
            
            # Calculate metrics
            response_time_ms = int((time.perf_counter() - start_time) * 1000)
            
            # Create analytics object
            analytics = QueryAnalytics(
                query_id=query_id,
                user_input=clean_query,
                ai_response=full_response,
                query_type=query_type,
                confidence=confidence,
                response_time_ms=response_time_ms,
                tokens_used=self._estimate_tokens(clean_query, full_response),
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                cost_estimate=self._calculate_cost_estimate()
            )
            
            # Update statistics
            self._update_session_stats(query_type, response_time_ms, analytics.tokens_used, True)
            
            # Add to history
            self.conversation_history.append(analytics)
            
            # Yield final analytics
            yield analytics
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            error_analytics = QueryAnalytics(
                query_id=query_id,
                user_input=user_query,
                ai_response="❌ An error occurred processing your query.",
                query_type="error",
                confidence=0.0,
                response_time_ms=0,
                tokens_used=0,
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S")
            )
            yield error_analytics
    
    def _estimate_tokens(self, input_text: str, output_text: str) -> int:
        """Estimate token usage (rough approximation)"""
        # Rough estimate: 1 token ≈ 4 characters for English text
        return (len(input_text) + len(output_text)) // 4
    
    def _calculate_cost_estimate(self) -> float:
        """Calculate cost estimate based on model and tokens"""
        pricing = {
            "gpt-3.5-turbo": 0.002 / 1000,  # $0.002 per 1K tokens
            "gpt-4": 0.03 / 1000,  # $0.03 per 1K tokens
        }
        
        base_cost = pricing.get(self.model, 0.002 / 1000)
        return self.session_stats["total_tokens_used"] * base_cost
    
    def _update_session_stats(self, query_type: str, response_time_ms: int, tokens_used: int, success: bool):
        """Update session statistics"""
        self.session_stats["queries_processed"] += 1
        
        if success:
            self.session_stats["successful_queries"] += 1
        
        self.session_stats["total_tokens_used"] += tokens_used
        
        # Update average response time
        total_time = (self.session_stats["average_response_time"] * 
                     (self.session_stats["queries_processed"] - 1) + response_time_ms)
        self.session_stats["average_response_time"] = total_time / self.session_stats["queries_processed"]
        
        # Update query type distribution
        if query_type in self.session_stats["query_types"]:
            self.session_stats["query_types"][query_type] += 1
        else:
            self.session_stats["query_types"][query_type] = 1
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        if not self.conversation_history:
            return {
                "total_queries": 0,
                "success_rate": 100.0,
                "average_confidence": 0.0,
                "average_response_time": 0,
                "total_tokens": 0,
                "query_distribution": {},
                "session_duration": 0
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
    
    def export_conversation_data(self) -> List[Dict[str, Any]]:
        """Export conversation history for analysis"""
        return [asdict(query) for query in self.conversation_history]


# Utility functions for compatibility and display
def format_response_for_display(response: str) -> str:
    """Format AI response for better display"""
    return response.replace("\n", "  \n")


def calculate_usage_cost(tokens_used: int, model: str = "gpt-3.5-turbo") -> float:
    """Calculate approximate API usage cost"""
    pricing = {
        "gpt-3.5-turbo": 0.002 / 1000,
        "gpt-3.5-turbo-16k": 0.004 / 1000,
        "gpt-4": 0.03 / 1000,
        "gpt-4-32k": 0.06 / 1000
    }
    
    return tokens_used * pricing.get(model, 0.002 / 1000)


# For backward compatibility, create an alias
AdvancedQuickQueryAgent = EnhancedQuickQueryAgent


if __name__ == "__main__":
    # Demo usage
    import asyncio
    
    async def main():
        agent = EnhancedQuickQueryAgent()
        
        # Test queries
        test_queries = [
            "When is the next exam?",
            "Tell me about the AI/ML syllabus",
            "What are the library timings?",
            "How do I contact the placement office?"
        ]
        
        print("🚀 Enhanced QuickQuery Agent Demo")
        print("=" * 50)
        
        for query in test_queries:
            print(f"\nUser: {query}")
            print("Assistant: ", end="")
            
            final_analytics = None
            async for result in agent.process_query_stream(query):
                if isinstance(result, str):
                    print(result, end="", flush=True)
                elif isinstance(result, QueryAnalytics):
                    final_analytics = result
            
            print()  # New line
            if final_analytics:
                print(f"[{final_analytics.query_type} | {final_analytics.confidence:.2f} | {final_analytics.response_time_ms}ms]")
        
        # Print session analytics
        analytics = agent.get_analytics_summary()
        print(f"\n📊 Session Summary:")
        print(f"Queries: {analytics['total_queries']} | Success: {analytics['success_rate']:.1f}%")
        print(f"Avg Response Time: {analytics['average_response_time']:.0f}ms")
    
    asyncio.run(main())