"""
Deployment and Production Utilities for QuickQuery
Health checks, monitoring, and deployment automation
"""

import os
import sys
import time
import psutil
import requests
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging

from config import get_config


class HealthChecker:
    """System health monitoring and checks"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
    
    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check"""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",
            "checks": {}
        }
        
        try:
            # System resources
            health_status["checks"]["cpu"] = self._check_cpu_usage()
            health_status["checks"]["memory"] = self._check_memory_usage()
            health_status["checks"]["disk"] = self._check_disk_usage()
            
            # Application checks
            health_status["checks"]["configuration"] = self._check_configuration()
            health_status["checks"]["dependencies"] = self._check_dependencies()
            health_status["checks"]["openai_api"] = self._check_openai_connection()
            
            # Determine overall status
            if any(check["status"] == "error" for check in health_status["checks"].values()):
                health_status["status"] = "unhealthy"
            elif any(check["status"] == "warning" for check in health_status["checks"].values()):
                health_status["status"] = "degraded"
        
        except Exception as e:
            health_status["status"] = "error"
            health_status["error"] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return health_status
    
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            return {
                "status": "error" if cpu_percent > 90 else "warning" if cpu_percent > 70 else "healthy",
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "message": f"CPU usage: {cpu_percent}%"
            }
        except Exception as e:
            return {"status": "error", "message": f"CPU check failed: {e}"}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            return {
                "status": "error" if memory.percent > 90 else "warning" if memory.percent > 80 else "healthy",
                "memory_percent": memory.percent,
                "memory_total": memory.total,
                "memory_available": memory.available,
                "message": f"Memory usage: {memory.percent}%"
            }
        except Exception as e:
            return {"status": "error", "message": f"Memory check failed: {e}"}
    
    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage"""
        try:
            disk = psutil.disk_usage('/')
            percent_used = (disk.used / disk.total) * 100
            return {
                "status": "error" if percent_used > 90 else "warning" if percent_used > 80 else "healthy",
                "disk_percent": percent_used,
                "disk_total": disk.total,
                "disk_free": disk.free,
                "message": f"Disk usage: {percent_used:.1f}%"
            }
        except Exception as e:
            return {"status": "error", "message": f"Disk check failed: {e}"}
    
    def _check_configuration(self) -> Dict[str, Any]:
        """Check application configuration"""
        try:
            issues = self.config.validate_config()
            
            if issues["errors"]:
                return {
                    "status": "error",
                    "errors": issues["errors"],
                    "warnings": issues["warnings"],
                    "message": f"Configuration has {len(issues['errors'])} errors"
                }
            elif issues["warnings"]:
                return {
                    "status": "warning",
                    "warnings": issues["warnings"],
                    "message": f"Configuration has {len(issues['warnings'])} warnings"
                }
            else:
                return {
                    "status": "healthy",
                    "message": "Configuration is valid"
                }
        except Exception as e:
            return {"status": "error", "message": f"Configuration check failed: {e}"}
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies"""
        try:
            required_packages = [
                "streamlit", "openai", "pandas", "plotly", 
                "python-dotenv", "requests", "psutil"
            ]
            
            missing_packages = []
            for package in required_packages:
                try:
                    __import__(package.replace("-", "_"))
                except ImportError:
                    missing_packages.append(package)
            
            if missing_packages:
                return {
                    "status": "error",
                    "missing_packages": missing_packages,
                    "message": f"Missing packages: {', '.join(missing_packages)}"
                }
            else:
                return {
                    "status": "healthy",
                    "message": "All dependencies available"
                }
        except Exception as e:
            return {"status": "error", "message": f"Dependency check failed: {e}"}
    
    def _check_openai_connection(self) -> Dict[str, Any]:
        """Check OpenAI API connection"""
        if not self.config.openai.api_key:
            return {
                "status": "warning",
                "message": "OpenAI API key not configured"
            }
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.config.openai.api_key)
            
            # Test with minimal request
            response = client.chat.completions.create(
                model=self.config.openai.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            
            return {
                "status": "healthy",
                "model": self.config.openai.model,
                "message": "OpenAI API connection successful"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"OpenAI API connection failed: {e}"
            }


class DeploymentManager:
    """Deployment automation and management"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        self.health_checker = HealthChecker(config)
    
    def deploy(self, target_environment: str = "production") -> Dict[str, Any]:
        """Deploy application to target environment"""
        deployment_result = {
            "timestamp": datetime.now().isoformat(),
            "environment": target_environment,
            "status": "in_progress",
            "steps": []
        }
        
        try:
            # Pre-deployment checks
            self._add_step(deployment_result, "pre_deployment_checks", "running")
            pre_check_result = self._run_pre_deployment_checks()
            
            if not pre_check_result["success"]:
                self._add_step(deployment_result, "pre_deployment_checks", "failed", pre_check_result["message"])
                deployment_result["status"] = "failed"
                return deployment_result
            
            self._add_step(deployment_result, "pre_deployment_checks", "completed")
            
            # Install dependencies
            self._add_step(deployment_result, "install_dependencies", "running")
            if self._install_dependencies():
                self._add_step(deployment_result, "install_dependencies", "completed")
            else:
                self._add_step(deployment_result, "install_dependencies", "failed")
                deployment_result["status"] = "failed"
                return deployment_result
            
            # Run tests
            self._add_step(deployment_result, "run_tests", "running")
            test_result = self._run_tests()
            if test_result["success"]:
                self._add_step(deployment_result, "run_tests", "completed", f"Passed {test_result['passed']} tests")
            else:
                self._add_step(deployment_result, "run_tests", "failed", test_result["message"])
                deployment_result["status"] = "failed"
                return deployment_result
            
            # Deploy application
            self._add_step(deployment_result, "deploy_application", "running")
            if self._deploy_application(target_environment):
                self._add_step(deployment_result, "deploy_application", "completed")
            else:
                self._add_step(deployment_result, "deploy_application", "failed")
                deployment_result["status"] = "failed"
                return deployment_result
            
            # Post-deployment verification
            self._add_step(deployment_result, "post_deployment_verification", "running")
            verification_result = self._run_post_deployment_verification()
            if verification_result["success"]:
                self._add_step(deployment_result, "post_deployment_verification", "completed")
                deployment_result["status"] = "completed"
            else:
                self._add_step(deployment_result, "post_deployment_verification", "failed", verification_result["message"])
                deployment_result["status"] = "failed"
            
        except Exception as e:
            deployment_result["status"] = "error"
            deployment_result["error"] = str(e)
            self.logger.error(f"Deployment failed: {e}")
        
        return deployment_result
    
    def _add_step(self, deployment_result: dict, step_name: str, status: str, message: str = ""):
        """Add deployment step to result"""
        deployment_result["steps"].append({
            "name": step_name,
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def _run_pre_deployment_checks(self) -> Dict[str, Any]:
        """Run pre-deployment checks"""
        try:
            health_result = self.health_checker.check_system_health()
            
            if health_result["status"] == "unhealthy":
                return {
                    "success": False,
                    "message": "System health check failed"
                }
            
            # Check git status (if in git repo)
            try:
                result = subprocess.run(["git", "status", "--porcelain"], 
                                      capture_output=True, text=True, timeout=10)
                if result.stdout.strip():
                    return {
                        "success": False,
                        "message": "Working directory has uncommitted changes"
                    }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # Not in git repo or git not available
                pass
            
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "message": f"Pre-deployment check failed: {e}"}
    
    def _install_dependencies(self) -> bool:
        """Install required dependencies"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True, timeout=300)
            
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Dependency installation failed: {e}")
            return False
    
    def _run_tests(self) -> Dict[str, Any]:
        """Run application tests"""
        try:
            # Check if test file exists
            test_file = Path("test_quickquery.py")
            if not test_file.exists():
                return {
                    "success": True,
                    "passed": 0,
                    "message": "No tests found"
                }
            
            result = subprocess.run([
                sys.executable, "-m", "pytest", str(test_file), "-v"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Parse test results (simplified)
                passed_tests = result.stdout.count("PASSED")
                return {
                    "success": True,
                    "passed": passed_tests,
                    "message": f"All tests passed"
                }
            else:
                return {
                    "success": False,
                    "message": f"Tests failed: {result.stderr}"
                }
                
        except Exception as e:
            return {"success": False, "message": f"Test execution failed: {e}"}
    
    def _deploy_application(self, environment: str) -> bool:
        """Deploy application to target environment"""
        try:
            if environment == "production":
                # Production deployment logic
                return self._deploy_to_production()
            elif environment == "staging":
                # Staging deployment logic
                return self._deploy_to_staging()
            else:
                # Local/development deployment
                return True
        except Exception as e:
            self.logger.error(f"Application deployment failed: {e}")
            return False
    
    def _deploy_to_production(self) -> bool:
        """Deploy to production environment"""
        # This would typically involve:
        # - Building Docker image
        # - Pushing to registry
        # - Updating Kubernetes/container orchestration
        # - Rolling deployment
        self.logger.info("Production deployment - implement based on your infrastructure")
        return True
    
    def _deploy_to_staging(self) -> bool:
        """Deploy to staging environment"""
        # Staging deployment logic
        self.logger.info("Staging deployment - implement based on your infrastructure")
        return True
    
    def _run_post_deployment_verification(self) -> Dict[str, Any]:
        """Run post-deployment verification"""
        try:
            # Health check after deployment
            health_result = self.health_checker.check_system_health()
            
            if health_result["status"] == "unhealthy":
                return {
                    "success": False,
                    "message": "Post-deployment health check failed"
                }
            
            # Additional verification checks can be added here
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "message": f"Post-deployment verification failed: {e}"}


class MonitoringSetup:
    """Setup monitoring and alerting"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
    
    def setup_monitoring(self) -> Dict[str, Any]:
        """Setup application monitoring"""
        setup_result = {
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        try:
            # Setup logging
            setup_result["components"]["logging"] = self._setup_logging()
            
            # Setup health check endpoint
            setup_result["components"]["health_checks"] = self._setup_health_checks()
            
            # Setup metrics collection
            setup_result["components"]["metrics"] = self._setup_metrics()
            
            return setup_result
            
        except Exception as e:
            setup_result["error"] = str(e)
            self.logger.error(f"Monitoring setup failed: {e}")
            return setup_result
    
    def _setup_logging(self) -> Dict[str, Any]:
        """Setup application logging"""
        try:
            # Ensure log directory exists
            if self.config.logging.file_path:
                log_dir = Path(self.config.logging.file_path).parent
                log_dir.mkdir(exist_ok=True, parents=True)
            
            return {"status": "configured", "file_path": self.config.logging.file_path}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _setup_health_checks(self) -> Dict[str, Any]:
        """Setup health check monitoring"""
        try:
            # Health check configuration
            return {
                "status": "configured",
                "endpoint": "/health",
                "interval": "60s"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup metrics collection"""
        try:
            # Metrics configuration
            return {
                "status": "configured",
                "enabled": self.config.app["enable_analytics"]
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


def main():
    """Main deployment script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QuickQuery Deployment Manager")
    parser.add_argument("action", choices=["health", "deploy", "monitor"], 
                       help="Action to perform")
    parser.add_argument("--environment", default="development",
                       help="Target environment")
    parser.add_argument("--output", default="console",
                       choices=["console", "json"],
                       help="Output format")
    
    args = parser.parse_args()
    
    config = get_config(args.environment)
    result = {"status": "unknown", "message": "No action performed"}
    
    if args.action == "health":
        health_checker = HealthChecker(config)
        result = health_checker.check_system_health()
    
    elif args.action == "deploy":
        deployment_manager = DeploymentManager(config)
        result = deployment_manager.deploy(args.environment)
    
    elif args.action == "monitor":
        monitoring_setup = MonitoringSetup(config)
        result = monitoring_setup.setup_monitoring()
    
    # Output results
    if args.output == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"\n🚀 QuickQuery {args.action.title()} - {args.environment.title()}")
        print("=" * 50)
        
        if "status" in result:
            status_emoji = {
                "healthy": "✅",
                "degraded": "⚠️",
                "unhealthy": "❌",
                "completed": "✅",
                "failed": "❌",
                "error": "❌"
            }.get(result["status"], "ℹ️")
            
            print(f"{status_emoji} Status: {result['status'].title()}")
        
        if args.action == "health" and "checks" in result and isinstance(result["checks"], dict):
            print("\n📊 Health Checks:")
            for check_name, check_result in result["checks"].items():
                if isinstance(check_result, dict):
                    status_emoji = {"healthy": "✅", "warning": "⚠️", "error": "❌"}
                    emoji = status_emoji.get(check_result.get("status", "unknown"), "ℹ️")
                    message = check_result.get("message", "No message")
                    print(f"  {emoji} {check_name}: {message}")
        
        elif args.action == "deploy" and "steps" in result and isinstance(result["steps"], list):
            print("\n📋 Deployment Steps:")
            for step in result["steps"]:
                if isinstance(step, dict):
                    status_emoji = {"completed": "✅", "running": "🔄", "failed": "❌"}
                    emoji = status_emoji.get(step.get("status", "unknown"), "ℹ️")
                    name = step.get("name", "Unknown step")
                    status = step.get("status", "unknown")
                    print(f"  {emoji} {name}: {status}")
                    if step.get("message"):
                        print(f"      {step['message']}")


if __name__ == "__main__":
    main()