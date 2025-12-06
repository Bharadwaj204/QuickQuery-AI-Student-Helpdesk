#!/usr/bin/env python3
"""
QuickQuery AI Student Helpdesk - Main Entry Point
Production-ready launcher with health checks and monitoring
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config import get_config
    from deployment import HealthChecker, DeploymentManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Warning: Configuration modules not available")


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        "streamlit",
        "openai", 
        "pandas",
        "plotly",
        "python-dotenv",
        "requests",
        "psutil"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies available")
    return True


def setup_environment():
    """Setup environment and configuration"""
    env_file = Path(".env")
    env_template = Path(".env.template")
    
    if not env_file.exists() and env_template.exists():
        print("📝 Creating .env file from template...")
        env_file.write_text(env_template.read_text())
        print("⚠️  Please edit .env file with your OpenAI API key")
        return False
    
    return True


def run_health_check():
    """Run system health check before starting"""
    if not CONFIG_AVAILABLE:
        print("⚠️  Skipping health check - config not available")
        return True
    
    try:
        from deployment import HealthChecker
        print("🔍 Running health check...")
        health_checker = HealthChecker()
        health_status = health_checker.check_system_health()
        
        if health_status["status"] == "unhealthy":
            print("❌ Health check failed!")
            for check_name, result in health_status["checks"].items():
                if result["status"] == "error":
                    print(f"  • {check_name}: {result['message']}")
            return False
        
        print("✅ Health check passed")
        return True
        
    except Exception as e:
        print(f"⚠️  Health check error: {e}")
        return True  # Continue anyway


def start_streamlit(app_file="app.py", port=8501):
    """Start the Streamlit application"""
    if not Path(app_file).exists():
        print(f"❌ Application file not found: {app_file}")
        available_apps = [f for f in Path(".").glob("*app*.py") if f.is_file()]
        if available_apps:
            app_file = str(available_apps[0])
            print(f"🔄 Using {app_file} instead")
        else:
            print("❌ No application files found")
            return False
    
    print(f"🚀 Starting QuickQuery on port {port}...")
    print(f"📱 Application will be available at: http://localhost:{port}")
    print("Press Ctrl+C to stop the application\n")
    
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", app_file,
            "--server.port", str(port),
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, check=True)
        return True
        
    except KeyboardInterrupt:
        print("\n👋 QuickQuery stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def main():
    """Main application launcher"""
    print("🤖 QuickQuery AI Student Helpdesk")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return 1
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Setup environment
    if not setup_environment():
        print("🛠️  Please configure .env file and run again")
        return 1
    
    # Run health check
    if not run_health_check():
        response = input("Health check failed. Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    # Start application
    success = start_streamlit()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())