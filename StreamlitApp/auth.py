import streamlit as st
import hashlib
import sqlite3
import os
from pathlib import Path

# Initialize database
def init_db():
    conn = sqlite3.connect('golf_analytics.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    ''')
    
    # Create shots table with user_id foreign key
    c.execute('''
        CREATE TABLE IF NOT EXISTS shots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            shot_type TEXT NOT NULL,
            carry INTEGER NOT NULL,
            club_speed REAL NOT NULL,
            ball_speed REAL NOT NULL,
            launch_angle REAL NOT NULL,
            spin_rate INTEGER NOT NULL,
            face_angle REAL NOT NULL,
            face_to_path REAL NOT NULL,
            club_path REAL NOT NULL,
            attack_angle REAL NOT NULL,
            launch_direction REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Register new user
def register_user(username, password, email):
    conn = sqlite3.connect('golf_analytics.db')
    c = conn.cursor()
    
    try:
        c.execute('''
            INSERT INTO users (username, password_hash, email)
            VALUES (?, ?, ?)
        ''', (username, hash_password(password), email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# Verify user login
def verify_user(username, password):
    conn = sqlite3.connect('golf_analytics.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT id, password_hash FROM users
        WHERE username = ?
    ''', (username,))
    
    result = c.fetchone()
    conn.close()
    
    if result and result[1] == hash_password(password):
        return result[0]  # Return user_id
    return None

# Get user's shots
def get_user_shots(user_id):
    conn = sqlite3.connect('golf_analytics.db')
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM shots
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (user_id,))
    
    shots = c.fetchall()
    conn.close()
    return shots

# Save user's shot
def save_user_shot(user_id, shot_data):
    conn = sqlite3.connect('golf_analytics.db')
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO shots (
            user_id, shot_type, carry, club_speed, ball_speed,
            launch_angle, spin_rate, face_angle, face_to_path,
            club_path, attack_angle, launch_direction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id,
        shot_data['Shot Type'],
        shot_data['Carry (yards)'],
        shot_data['Club Speed (MPH)'],
        shot_data['Ball Speed (MPH)'],
        shot_data['Launch Angle (Deg)'],
        shot_data['Spin Rate (RPM)'],
        shot_data['Face Angle (Deg)'],
        shot_data['Face to Path (Deg)'],
        shot_data['Club Path (Deg)'],
        shot_data['Attack Angle (Deg)'],
        shot_data['Launch Direction (Deg)']
    ))
    
    conn.commit()
    conn.close()

# Delete user's shot
def delete_user_shot(shot_id):
    conn = sqlite3.connect('golf_analytics.db')
    c = conn.cursor()
    
    c.execute('''
        DELETE FROM shots
        WHERE id = ?
    ''', (shot_id,))
    
    conn.commit()
    conn.close()

# Streamlit UI for authentication
def show_auth_page():
    st.title("Golf Analytics - Authentication")
    
    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    # Initialize database
    init_db()
    
    # Login/Signup tabs
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            user_id = verify_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        st.subheader("Sign Up")
        new_username = st.text_input("Username", key="signup_username")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Sign Up"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                if register_user(new_username, new_password, new_email):
                    st.success("Registration successful! Please login.")
                else:
                    st.error("Username or email already exists")
    
    # Logout button if logged in
    if st.session_state.user_id:
        if st.button("Logout"):
            st.session_state.user_id = None
            st.rerun() 