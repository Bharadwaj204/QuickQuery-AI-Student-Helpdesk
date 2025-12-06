"""
Production Configuration Management for QuickQuery
Environment-based configuration with security best practices
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    name: str = "quickquery"
    user: str = "quickquery_user"
    password: str = ""
    ssl_mode: str = "prefer"


@dataclass
class OpenAIConfig:
    """OpenAI API configuration"""
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3


@dataclass
class SecurityConfig:
    """Security configuration settings"""
    secret_key: str = ""
    session_timeout: int = 3600  # 1 hour
    max_request_size: int = 1048576  # 1MB
    rate_limit_per_minute: int = 60
    allowed_origins: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = ["http://localhost:8501", "https://*.streamlit.app"]


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5


class Config:
    """Main configuration class with environment-based settings"""
    
    def __init__(self, environment: Optional[str] = None):
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.debug = self.environment == "development"
        
        # Load configurations
        self.app = self._load_app_config()
        self.openai = self._load_openai_config()
        self.security = self._load_security_config()
        self.logging = self._load_logging_config()
        self.database = self._load_database_config()
        
        # Setup logging
        self._setup_logging()
    
    def _load_app_config(self) -> Dict[str, Any]:
        """Load application configuration"""
        return {
            "title": os.getenv("APP_TITLE", "QuickQuery - AI Student Helpdesk"),
            "description": os.getenv("APP_DESCRIPTION", "Enhanced AI-powered student assistance"),
            "version": os.getenv("APP_VERSION", "2.0.0"),
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8501")),
            "max_chat_history": int(os.getenv("MAX_CHAT_HISTORY", "50")),
            "enable_analytics": os.getenv("ENABLE_ANALYTICS", "true").lower() == "true",
            "enable_feedback": os.getenv("ENABLE_FEEDBACK", "true").lower() == "true",
            "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),  # 1 hour
        }
    
    def _load_openai_config(self) -> OpenAIConfig:
        """Load OpenAI configuration"""
        return OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "500")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "30")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        )
    
    def _load_security_config(self) -> SecurityConfig:
        """Load security configuration"""
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else None
        
        return SecurityConfig(
            secret_key=os.getenv("SECRET_KEY", self._generate_secret_key()),
            session_timeout=int(os.getenv("SESSION_TIMEOUT", "3600")),
            max_request_size=int(os.getenv("MAX_REQUEST_SIZE", "1048576")),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            allowed_origins=allowed_origins
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration"""
        log_file = None
        if os.getenv("LOG_TO_FILE", "false").lower() == "true":
            log_dir = Path(os.getenv("LOG_DIR", "logs"))
            log_dir.mkdir(exist_ok=True)
            log_file = str(log_dir / f"quickquery_{self.environment}.log")
        
        return LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=log_file,
            max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", "10485760")),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration"""
        return DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "quickquery"),
            user=os.getenv("DB_USER", "quickquery_user"),
            password=os.getenv("DB_PASSWORD", ""),
            ssl_mode=os.getenv("DB_SSL_MODE", "prefer")
        )
    
    def _generate_secret_key(self) -> str:
        """Generate a secret key for development"""
        import secrets
        return secrets.token_hex(32)
    
    def _setup_logging(self):
        """Setup application logging"""
        logging.basicConfig(
            level=getattr(logging, self.logging.level.upper()),
            format=self.logging.format
        )
        
        # Add file handler if specified
        if self.logging.file_path:
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.max_file_size,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(logging.Formatter(self.logging.format))
            
            logger = logging.getLogger()
            logger.addHandler(file_handler)
    
    def validate_config(self) -> Dict[str, list]:
        """Validate configuration and return any issues"""
        issues = {
            "errors": [],
            "warnings": []
        }
        
        # Critical validations
        if not self.openai.api_key and self.environment == "production":
            issues["errors"].append("OPENAI_API_KEY is required for production")
        
        if not self.security.secret_key:
            issues["warnings"].append("SECRET_KEY not set, using generated key")
        
        if self.openai.max_tokens <= 0:
            issues["errors"].append("OPENAI_MAX_TOKENS must be positive")
        
        if not 0 <= self.openai.temperature <= 2:
            issues["errors"].append("OPENAI_TEMPERATURE must be between 0 and 2")
        
        if self.security.rate_limit_per_minute <= 0:
            issues["errors"].append("RATE_LIMIT_PER_MINUTE must be positive")
        
        # Environment-specific validations
        if self.environment == "production":
            if self.debug:
                issues["warnings"].append("Debug mode is enabled in production")
            
            if self.security.allowed_origins and "localhost" in self.security.allowed_origins:
                issues["warnings"].append("Localhost is allowed in production CORS")
        
        return issues
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration"""
        return {
            "environment": self.environment,
            "debug": self.debug,
            "app_version": self.app["version"],
            "openai_model": self.openai.model,
            "security_enabled": bool(self.security.secret_key),
            "logging_level": self.logging.level,
            "database_host": self.database.host,
            "features": {
                "analytics": self.app["enable_analytics"],
                "feedback": self.app["enable_feedback"]
            }
        }
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Export configuration for debugging (optionally including secrets)"""
        config_dict = {
            "environment": self.environment,
            "app": self.app,
            "openai": {
                "model": self.openai.model,
                "max_tokens": self.openai.max_tokens,
                "temperature": self.openai.temperature,
                "timeout": self.openai.timeout
            },
            "security": {
                "session_timeout": self.security.session_timeout,
                "max_request_size": self.security.max_request_size,
                "rate_limit_per_minute": self.security.rate_limit_per_minute,
                "allowed_origins": self.security.allowed_origins
            },
            "logging": {
                "level": self.logging.level,
                "file_path": self.logging.file_path
            }
        }
        
        if include_secrets:
            config_dict["openai"]["api_key"] = self.openai.api_key[:10] + "..." if self.openai.api_key else None
            config_dict["security"]["secret_key"] = bool(self.security.secret_key)
            config_dict["database"] = {
                "host": self.database.host,
                "port": self.database.port,
                "name": self.database.name,
                "user": self.database.user,
                "password_set": bool(self.database.password)
            }
        
        return config_dict


# Environment-specific configurations
class DevelopmentConfig(Config):
    """Development environment configuration"""
    
    def __init__(self):
        super().__init__("development")
        self.debug = True


class ProductionConfig(Config):
    """Production environment configuration"""
    
    def __init__(self):
        super().__init__("production")
        self.debug = False
        
        # Override with production-specific settings
        if not self.openai.api_key:
            raise ValueError("OPENAI_API_KEY is required for production")


class TestingConfig(Config):
    """Testing environment configuration"""
    
    def __init__(self):
        super().__init__("testing")
        self.debug = True
        
        # Override with testing-specific settings
        self.app["enable_analytics"] = False
        self.openai.api_key = "test-key"


# Configuration factory
def get_config(environment: Optional[str] = None) -> Config:
    """Get configuration based on environment"""
    env = environment or os.getenv("ENVIRONMENT", "development")
    
    config_classes = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig
    }
    
    config_class = config_classes.get(env, Config)
    return config_class()


# Global configuration instance
config = get_config()


if __name__ == "__main__":
    # Configuration validation and testing
    print("🔧 QuickQuery Configuration Summary")
    print("=" * 50)
    
    config = get_config()
    
    # Print configuration summary
    summary = config.get_config_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n🔍 Configuration Validation")
    print("-" * 30)
    
    issues = config.validate_config()
    
    if issues["errors"]:
        print("❌ Errors:")
        for error in issues["errors"]:
            print(f"  • {error}")
    
    if issues["warnings"]:
        print("⚠️ Warnings:")
        for warning in issues["warnings"]:
            print(f"  • {warning}")
    
    if not issues["errors"] and not issues["warnings"]:
        print("✅ Configuration is valid!")
    
    print(f"\n📊 Total Config Items: {len(config.export_config())}")
    print(f"🛡️ Security Level: {'High' if config.environment == 'production' else 'Development'}")