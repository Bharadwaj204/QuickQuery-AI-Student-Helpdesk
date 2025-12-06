# QuickQuery AI Student Helpdesk

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/openai-1.12+-green.svg)](https://openai.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An advanced, production-ready AI-powered student helpdesk application built with Streamlit and OpenAI. This application provides intelligent campus assistance with real-time analytics, modern UI, and robust error handling.

## 🌟 Key Features

### 🤖 AI-Powered Assistance
- **Modern OpenAI API v1+ integration** with support for multiple models (GPT-3.5, GPT-4)
- **Real-time streaming responses** for interactive user experience
- **Intelligent query classification** to categorize and route questions appropriately
- **Context-aware conversations** with persistent chat history
- **Fallback to knowledge base** when API is unavailable or quota exceeded

### 📊 Real-Time Analytics
- **Live system metrics dashboard** with performance indicators
- **Query distribution visualization** to understand usage patterns
- **Response time tracking** for performance monitoring
- **Success rate monitoring** to ensure service reliability
- **Usage cost estimation** to track API consumption

### 🎨 Modern UI/UX
- **Card-based responsive design** for optimal viewing on all devices
- **Dark/light theme support** with automatic system preference detection
- **Smooth animations and transitions** for enhanced user experience
- **Professional styling with gradients** and modern aesthetics
- **Mobile-friendly layout** for on-the-go access

### 🔒 Security & Reliability
- **Input sanitization and validation** to prevent injection attacks
- **API key management** with secure storage
- **Rate limiting protection** to prevent abuse
- **Error handling and graceful degradation** for robust operation
- **Session management** with configurable timeouts

## 🛠️ Technology Stack

### Core Framework
- **Python 3.8+** - Primary programming language
- **Streamlit 1.28+** - Web framework for data applications
- **OpenAI SDK 1.12+** - Official OpenAI API client

### Data Processing & Visualization
- **Pandas 2.0+** - Data manipulation and analysis
- **Plotly 5.15+** - Interactive data visualization
- **NumPy 1.24+** - Numerical computing

### Infrastructure & Utilities
- **AIOHTTP 3.8+** - Asynchronous HTTP client/server
- **Requests 2.31+** - HTTP library for synchronous requests
- **Python-Dotenv 1.0+** - Environment variable management
- **PSUtil 5.9+** - System and process utilities

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key (get yours at [platform.openai.com](https://platform.openai.com/api-keys))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd QuickQuery-AI-Student-Helpdesk
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.template .env
# Edit .env with your OpenAI API key and other settings
```

4. Run the application:
```bash
streamlit run app.py
```

## ⚙️ Configuration

Create a `.env` file with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Application Settings
ENVIRONMENT=development
DEBUG=true
ENABLE_ANALYTICS=true
MAX_CHAT_HISTORY=20

# Security
SECRET_KEY=your_secret_key_here_generate_a_strong_random_key

# Logging
LOG_LEVEL=INFO
```

### Environment Variables Reference

| Variable | Description | Default |
|---------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | `""` |
| `OPENAI_MODEL` | OpenAI model to use | `"gpt-3.5-turbo"` |
| `ENVIRONMENT` | Environment mode | `"development"` |
| `DEBUG` | Enable debug mode | `true` |
| `ENABLE_ANALYTICS` | Enable analytics dashboard | `true` |
| `MAX_CHAT_HISTORY` | Maximum chat history to retain | `20` |
| `SECRET_KEY` | Secret key for security | Auto-generated |
| `LOG_LEVEL` | Logging level | `"INFO"` |

## 📱 Usage

1. Access the application at `http://localhost:8501`
2. Configure your OpenAI API key in the sidebar
3. Ask questions about:
   - Academic information (exams, syllabus, assignments)
   - Campus facilities (library, dining, sports)
   - Events and opportunities
   - Contact information

### Sample Queries
- "When are the next exams?"
- "What's on the AI/ML syllabus?"
- "Are there any upcoming events?"
- "What are the library hours?"

## 🏗️ Architecture

### Core Components

- **EnhancedQuickQueryAgent**: Main AI agent with streaming capabilities and intelligent query processing
- **KnowledgeBase**: Structured campus information storage with caching
- **SecurityValidator**: Input sanitization and validation utilities
- **QueryAnalytics**: Performance metrics collection and reporting
- **ConfigurationManager**: Environment-based configuration with validation

### Data Flow
```
User Query 
    ↓
Classification Engine 
    ↓
Context Retrieval 
    ↓
AI/Knowledge Base Processing 
    ↓
Response Streaming 
    ↓
Analytics Collection
```

### System Design Principles

1. **Modular Architecture**: Clean separation of concerns with well-defined interfaces
2. **Configuration-Driven**: All settings managed through environment variables
3. **Security First**: Input validation, sanitization, and secure credential handling
4. **Observability**: Comprehensive logging and metrics collection
5. **Resilience**: Graceful degradation and fallback mechanisms

## 📁 Project Structure

```
QuickQuery-AI-Student-Helpdesk/
├── app.py                     # Main Streamlit application
├── advanced_agent.py           # AI agent and business logic
├── config.py                   # Configuration management
├── deployment.py               # Deployment utilities
├── run.py                     # Application launcher
├── auth.py                    # Authentication utilities
├── requirements.txt            # Python dependencies
├── .env.template               # Environment configuration template
└── README.md                  # Project documentation
```

### Component Descriptions

- **app.py**: Main Streamlit application with UI components and user interaction handling
- **advanced_agent.py**: Core AI logic including query processing, classification, and response generation
- **config.py**: Configuration management with environment variable parsing and validation
- **deployment.py**: Production utilities including health checks and deployment helpers
- **run.py**: Application launcher with dependency checking and health monitoring
- **auth.py**: Authentication and authorization utilities

## 🧪 Testing

Run the built-in demo:
```bash
python run.py
```

This will perform system health checks and start the application with proper error handling.

## ☁️ Deployment

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" and select your repository
4. Set the main file path to `app.py`
5. Add your OpenAI API key in the Secrets section

### Manual Deployment

For production deployment:

1. Set `ENVIRONMENT=production` in `.env`
2. Use a production-grade WSGI server
3. Configure proper SSL certificates
4. Set up monitoring and alerting
5. Implement database persistence (optional)

### Docker Deployment (Coming Soon)

Containerized deployment options will be available in future releases.

## 🤝 Contributing

We welcome contributions to the QuickQuery AI Student Helpdesk project!

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 coding standards
- Write clear, descriptive commit messages
- Include tests for new functionality
- Update documentation as needed
- Maintain backward compatibility when possible

### Reporting Issues

- Use the GitHub issue tracker
- Provide detailed reproduction steps
- Include screenshots when relevant
- Specify your environment (OS, Python version, etc.)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For issues and feature requests, please create an issue in the repository.

### Community Resources

- [Documentation](docs/)
- [Issue Tracker](issues/)
- [Discussion Forum](discussions/)

## 🎯 Future Enhancements

Planned improvements include:
- Database integration for persistent storage
- Multi-language support
- Advanced authentication systems
- Customizable knowledge base
- Mobile application
- Voice input/output capabilities

---

*Built with ❤️ for students everywhere*