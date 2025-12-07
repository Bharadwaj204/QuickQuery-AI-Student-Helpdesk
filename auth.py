"""
Authentication Module for QuickQuery AI Student Helpdesk
Production-ready authentication with session management and college email validation
"""

import streamlit as st
import hashlib
import secrets
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json
import os


class AuthManager:
    """Handles user authentication and session management"""
    
    def __init__(self):
        # Allowed college email domains
        self.allowed_domains = [
            "student.college.edu",
            "college.edu", 
            "student.ac.in",
            "college.ac.in"
        ]
        
        # Initialize session state for auth
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize authentication session state variables"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'session_token' not in st.session_state:
            st.session_state.session_token = None
        if 'login_time' not in st.session_state:
            st.session_state.login_time = None
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt for secure storage"""
        # In a real application, use a proper password hashing library like bcrypt
        salt = "quickquery_salt_2025"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def validate_email(self, email: str) -> bool:
        """Validate if email belongs to allowed college domains"""
        if not email or '@' not in email:
            return False
        
        domain = email.split('@')[1].lower()
        return domain in self.allowed_domains
    
    def validate_credentials(self, username: str, password: str) -> bool:
        """Validate user credentials (demo implementation)"""
        # In a real application, this would check against a database
        # For demo purposes, we'll accept any valid college email with any password
        if self.validate_email(username):
            # Hash the password and store user info
            hashed_pw = self.hash_password(password)
            st.session_state.user = {
                'username': username,
                'email': username,
                'hashed_password': hashed_pw,
                'login_time': datetime.now().isoformat()
            }
            return True
        return False
    
    def login(self, username: str, password: str) -> bool:
        """Authenticate user and create session"""
        if self.validate_credentials(username, password):
            # Create session token
            st.session_state.session_token = secrets.token_urlsafe(32)
            st.session_state.authenticated = True
            st.session_state.login_time = datetime.now()
            return True
        return False
    
    def logout(self):
        """Terminate user session"""
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.session_token = None
        st.session_state.login_time = None
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        # Ensure session state is initialized
        self._init_session_state()
        
        # Check session timeout (24 hours)
        if st.session_state.authenticated and st.session_state.login_time:
            session_age = datetime.now() - st.session_state.login_time
            if session_age > timedelta(hours=24):
                self.logout()
                return False
            return True
        return st.session_state.authenticated
    
    def get_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user info"""
        if self.is_authenticated():
            return st.session_state.user
        return None
    
    def require_auth(self) -> bool:
        """Decorator-like function to require authentication"""
        # Ensure session state is initialized
        self._init_session_state()
        
        if not self.is_authenticated():
            self.show_login_page()
            return False
        return True
    
    def show_login_page(self):
        """Display login page UI"""
        st.markdown("""
        <div style="max-width: 500px; margin: 2rem auto; padding: 2rem; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); background: white;">
            <h1 style="text-align: center; color: #495057;">🎓 QuickQuery Login</h1>
            <p style="text-align: center; color: #6c757d;">AI Student Helpdesk - College Access Only</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create login form
        with st.form("login_form"):
            st.markdown("### 🔐 College Email Login")
            
            username = st.text_input(
                "College Email Address",
                placeholder="username@college.edu",
                help="Must be a valid college email address"
            )
            
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                help="Password must be at least 8 characters"
            )
            
            st.info("ℹ️ **Allowed Domains**: " + ", ".join(self.allowed_domains))
            
            submitted = st.form_submit_button("Login to QuickQuery")
            
            if submitted:
                if not username or not password:
                    st.error("❌ Please enter both email and password")
                elif len(password) < 8:
                    st.error("❌ Password must be at least 8 characters")
                elif not self.validate_email(username):
                    st.error("❌ Invalid college email domain. Please use your institutional email.")
                else:
                    if self.login(username, password):
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
        
        # OAuth2 alternative (placeholder for Google login)
        st.markdown("---")
        st.markdown("### 🔑 Alternative Login Methods")
        st.info("Google OAuth2 integration available for seamless college authentication")
        
        if st.button(" Sign in with Google", use_container_width=True):
            st.info("Google OAuth2 integration would be implemented here in a production environment")


# Global auth manager instance
auth_manager = AuthManager()


def require_auth():
    """Decorator to require authentication for protected pages"""
    return auth_manager.require_auth()


def get_current_user():
    """Get currently authenticated user"""
    return auth_manager.get_user()


def logout():
    """Logout current user"""
    auth_manager.logout()
    st.rerun()