# Create enhanced requirements file
enhanced_requirements = """streamlit>=1.28.0
openai>=0.28.0
pandas>=2.0.0
plotly>=5.15.0
python-dotenv>=1.0.0
requests>=2.31.0
numpy>=1.24.0
"""

with open('enhanced_requirements.txt', 'w') as f:
    f.write(enhanced_requirements)

# Create configuration file for API keys and settings
config_file = """import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    \"\"\"Configuration settings for QuickQuery application\"\"\"
    
    # OpenAI API Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    OPENAI_MAX_TOKENS = int(os.getenv('OPENAI_MAX_TOKENS', '300'))
    OPENAI_TEMPERATURE = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
    
    # Application Settings
    APP_TITLE = "QuickQuery - Advanced AI Student Helpdesk"
    APP_DESCRIPTION = "Powered by OpenAI GPT | OpenAI × NxtWave Buildathon 2025"
    MAX_CHAT_HISTORY = int(os.getenv('MAX_CHAT_HISTORY', '50'))
    
    # UI Settings
    THEME_COLOR = "#1f77b4"
    SUCCESS_COLOR = "#28a745"
    ERROR_COLOR = "#dc3545"
    WARNING_COLOR = "#ffc107"
    
    # Analytics Settings
    ENABLE_ANALYTICS = os.getenv('ENABLE_ANALYTICS', 'true').lower() == 'true'
    ENABLE_LOGGING = os.getenv('ENABLE_LOGGING', 'true').lower() == 'true'
    
    @classmethod
    def get_openai_config(cls):
        \"\"\"Get OpenAI configuration dictionary\"\"\"
        return {
            'api_key': cls.OPENAI_API_KEY,
            'model': cls.OPENAI_MODEL,
            'max_tokens': cls.OPENAI_MAX_TOKENS,
            'temperature': cls.OPENAI_TEMPERATURE
        }
    
    @classmethod
    def validate_config(cls):
        \"\"\"Validate configuration settings\"\"\"
        issues = []
        
        if not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY is not set")
        
        if cls.OPENAI_MAX_TOKENS <= 0:
            issues.append("OPENAI_MAX_TOKENS must be positive")
        
        if not 0 <= cls.OPENAI_TEMPERATURE <= 2:
            issues.append("OPENAI_TEMPERATURE must be between 0 and 2")
        
        return issues
"""

with open('config.py', 'w') as f:
    f.write(config_file)

# Create environment template file
env_template = """# QuickQuery Configuration File
# Copy this to .env and fill in your values

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=300
OPENAI_TEMPERATURE=0.7

# Application Settings
MAX_CHAT_HISTORY=50
ENABLE_ANALYTICS=true
ENABLE_LOGGING=true

# Deployment Settings (for production)
PORT=8501
HOST=0.0.0.0
"""

with open('.env.template', 'w') as f:
    f.write(env_template)

print("✅ Enhanced configuration files created:")
print("  - enhanced_requirements.txt (with all dependencies)")
print("  - config.py (configuration management)")
print("  - .env.template (environment variables template)")