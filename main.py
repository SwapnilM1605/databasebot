from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify,abort
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import os
from functools import wraps
import pandas as pd
import io
import psycopg2
import logging
from werkzeug.utils import secure_filename
from database import db, init_db
from models import User, AdminLog, SharedDatabase, ChatThread, ChatMessage, ConfigSetting, Enterprise, AdminUser, Plan, DataDictionary
from config import Config
from datetime import datetime, timedelta
import logging
from sqlalchemy import text, create_engine
import bcrypt
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from llm_service import LLMService
from chat_routes import chat_bp
import requests
import re
import secrets
import json
from file_upload import create_table_from_file, sanitize_db_name
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import sqlite3
from pathlib import Path
import mysql.connector
from file_upload import file_upload_bp
import pyodbc 

from forecast import ForecastingService

# Initialize the Flask application
app = Flask(__name__)
app.config.from_object('config.Config')
Config.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User loader callback
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize database
init_db(app)

# Setup MySQL client engine
client_engine = create_engine(app.config['MYSQL_CLIENT_URI'])

# Ensure SECRET_KEY is set
if not app.config.get('SECRET_KEY'):
    app.logger.error("SECRET_KEY is not set in environment or config. Using a temporary key.")
    app.secret_key = secrets.token_hex(24)
else:
    app.secret_key = app.config['SECRET_KEY']

app.register_blueprint(chat_bp)
app.register_blueprint(file_upload_bp)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize LLM Service
try:
    llm_service = LLMService()
except Exception as e:
    logger.error(f"Failed to initialize LLMService: {str(e)}")
    llm_service = None

# Ensure folders exist
for folder in [app.config['REPORT_FOLDER'], app.config['UPLOAD_FOLDER'], app.config['ENTERPRISE_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# MySQL config file
MYSQL_CONFIG_FILE = os.path.join(app.config['UPLOAD_FOLDER'], 'mysql_config.json')
POSTGRES_CONFIG_FILE = os.path.join(app.config['UPLOAD_FOLDER'], 'postgres_config.json')
# main.py


# Update folder configuration for SQL Server
SQLSERVER_CONFIG_FILE = os.path.join(app.config['UPLOAD_FOLDER'], 'sqlserver_config.json')
ENTERPRISE_FOLDER = app.config['ENTERPRISE_FOLDER']

# Initialize Forecasting Service
try:
    forecasting_service = ForecastingService()
except Exception as e:
    logger.error(f"Failed to initialize ForecastingService: {str(e)}")
    forecasting_service = None

def load_db_configs():
    try:
        configs = {}
        if os.path.exists(MYSQL_CONFIG_FILE):
            with open(MYSQL_CONFIG_FILE, 'r') as f:
                configs['mysql'] = json.load(f)
                app.config['MYSQL_HOST'] = configs['mysql'].get('host')
                app.config['MYSQL_PORT'] = configs['mysql'].get('port', 3306)
                app.config['MYSQL_ADMIN_USER'] = configs['mysql'].get('user')
                app.config['MYSQL_ADMIN_PASSWORD'] = configs['mysql'].get('password')
        if os.path.exists(POSTGRES_CONFIG_FILE):
            with open(POSTGRES_CONFIG_FILE, 'r') as f:
                configs['postgres'] = json.load(f)
                app.config['POSTGRES_HOST'] = configs['postgres'].get('host')
                app.config['POSTGRES_PORT'] = configs['postgres'].get('port', 5432)
                app.config['POSTGRES_ADMIN_USER'] = configs['postgres'].get('user')
                app.config['POSTGRES_ADMIN_PASSWORD'] = configs['postgres'].get('password')
        if os.path.exists(SQLSERVER_CONFIG_FILE):
            with open(SQLSERVER_CONFIG_FILE, 'r') as f:
                configs['sqlserver'] = json.load(f)
                app.config['SQLSERVER_HOST'] = configs['sqlserver'].get('host')
                app.config['SQLSERVER_PORT'] = configs['sqlserver'].get('port', 1433)
                app.config['SQLSERVER_ADMIN_USER'] = configs['sqlserver'].get('user')
                app.config['SQLSERVER_ADMIN_PASSWORD'] = configs['sqlserver'].get('password')
        return configs if configs else None
    except Exception as e:
        logger.error(f"Error loading database configs: {str(e)}")
        return None

# Call this when app starts to load any existing config
load_db_configs()

# Base directory
basedir = Path(__file__).parent.parent

# Initialize default settings
with app.app_context():
    try:
        if not ConfigSetting.query.filter_by(key='SIGNUP_SECRET_KEY').first():
            default_secret = os.getenv('SIGNUP_SECRET_KEY', 'GKLife')
            new_setting = ConfigSetting(key='SIGNUP_SECRET_KEY', value=default_secret)
            db.session.add(new_setting)
            db.session.commit()
            logger.debug("Default settings initialized")
    except Exception as e:
        logger.error(f"Failed to initialize default settings: {str(e)}")
        raise

def get_current_enterprise():
    """Get the current enterprise from session"""
    if 'enterprise_id' not in session:
        return None
    return Enterprise.query.get(session['enterprise_id'])

def require_enterprise_auth(f):
    """Decorator to ensure enterprise authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'enterprise_id' not in session:
            flash("Please log into an enterprise first", 'error')
            return redirect(url_for('verify_secret'))
        return f(*args, **kwargs)
    return decorated_function

def load_shared_databases():
    """Load shared databases for current enterprise"""
    enterprise = get_current_enterprise()
    if not enterprise:
        return []
    return [{"name": db.database_name, "type": db.database_type} for db in enterprise.shared_databases]

@app.route('/api/shared-databases', methods=['GET'])
def get_shared_databases():
    try:
        shared_databases = load_shared_databases()
        return jsonify({"shared_databases": shared_databases})
    except Exception as e:
        logger.error(f"Error fetching shared databases: {str(e)}")
        return jsonify({"error": str(e)}), 500



# Define the log_admin_action function
def log_admin_action(username, action, action_description=None, enterprise_id=None):
    try:
        # Get enterprise_id from session if not provided
        if enterprise_id is None and 'enterprise_id' in session:
            enterprise_id = session['enterprise_id']
        
        if enterprise_id is None:
            # Try to get enterprise_id from the user
            user = User.query.filter_by(username=username).first()
            if user and user.enterprise_id:
                enterprise_id = user.enterprise_id
        
        if enterprise_id is None:
            raise ValueError("Enterprise ID is required for audit logging")
            
        log_entry = AdminLog(
            username=username,
            action=action,
            action_description=action_description,
            timestamp=datetime.now(),
            enterprise_id=enterprise_id
        )
        db.session.add(log_entry)
        db.session.commit()
    except Exception as e:
        logger.error(f"Error logging admin action: {e}")
        db.session.rollback()
        raise  # Re-raise the exception if you want calling code to handle it

# Function to create MySQL user and database
def create_mysql_user_and_db(username, password):
    try:
        enterprise = get_current_enterprise()
        if not enterprise or not enterprise.mysql_host:
            raise ValueError("MySQL configuration not found. Please configure MySQL connection first.")

        connection_string = f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}@{enterprise.mysql_host}:{enterprise.mysql_port}"
        engine = create_engine(connection_string)

        db_name = sanitize_db_name(username)
        logger.debug(f"Creating database and user for {username}, db_name: {db_name}")

        with engine.connect() as conn:
            # Create database if not exists
            conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
            
            # Create user if not exists
            escaped_username = username.replace("'", "''")
            escaped_password = password.replace("'", "''")
            
            conn.execute(text(
                f"CREATE USER IF NOT EXISTS '{escaped_username}'@'%' "
                f"IDENTIFIED BY '{escaped_password}'"
            ))
            conn.execute(text(
                f"CREATE USER IF NOT EXISTS '{escaped_username}'@'localhost' "
                f"IDENTIFIED BY '{escaped_password}'"
            ))
            
            # Grant all privileges on the user's personal database
            conn.execute(text(
                f"GRANT ALL PRIVILEGES ON `{db_name}`.* TO '{escaped_username}'@'%'"
            ))
            conn.execute(text(
                f"GRANT ALL PRIVILEGES ON `{db_name}`.* TO '{escaped_username}'@'localhost'"
            ))
            
            # For shared databases, grant SELECT privileges only
            shared_dbs = SharedDatabase.query.filter_by(
                enterprise_id=enterprise.id,
                database_type='mysql'
            ).all()
            
            for shared_db in shared_dbs:
                conn.execute(text(
                    f"GRANT SELECT ON `{shared_db.database_name}`.* TO '{escaped_username}'@'%'"
                ))
                conn.execute(text(
                    f"GRANT SELECT ON `{shared_db.database_name}`.* TO '{escaped_username}'@'localhost'"
                ))
            
            conn.execute(text("FLUSH PRIVILEGES"))

        return db_name, password

    except Exception as e:
        logger.error(f"Error creating MySQL user or database for {username}: {str(e)}")
        # Attempt to clean up if partial creation occurred
        try:
            with engine.connect() as conn:
                conn.execute(text(f"DROP DATABASE IF EXISTS `{db_name}`"))
                conn.execute(text(f"DROP USER IF EXISTS '{escaped_username}'@'%'"))
                conn.execute(text(f"DROP USER IF EXISTS '{escaped_username}'@'localhost'"))
                conn.execute(text("FLUSH PRIVILEGES"))
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed after error: {str(cleanup_error)}")
        
        raise ValueError(f"Failed to create MySQL user and database: {str(e)}")


@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/input_page')
def input():
    return render_template('inputs.html')

@app.route('/workspace')
def workspace():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('workspace.html')

@app.route('/pricing')
def pricing():
    return render_template('plans.html')

@app.route('/enterprise')
def enterprise():
    return render_template('enterprise.html')

@app.route('/create_enterprise', methods=['POST'])
def create_enterprise():
    try:
        # Get form data
        enterprise_name = request.form['enterprise_name']
        username = request.form['username']
        password = request.form['password']
        
        # Validate inputs
        if not all([enterprise_name, username, password]):
            flash('All fields are required', 'error')
            return redirect(url_for('enterprise'))
            
        if len(password) < 8:
            flash('Password must be at least 8 characters', 'error')
            return redirect(url_for('enterprise'))
            
        # Check if plan exists for the email (username)
        plan = Plan.query.filter_by(email=username).first()
        if not plan:
            flash('No plan found for this email. Please select a plan first.', 'error')
            return redirect(url_for('enterprise'))
            
        # Check if enterprise name already exists
        existing_enterprise = Enterprise.query.filter_by(name=enterprise_name).first()
        if existing_enterprise:
            flash('Enterprise name already exists', 'error')
            return redirect(url_for('enterprise'))
            
        # Check if admin username already exists
        existing_admin = AdminUser.query.filter_by(username=username).first()
        if existing_admin:
            flash('Admin username already exists', 'error')
            return redirect(url_for('enterprise'))
            
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Create enterprise in SQLite database
        enterprise = Enterprise(name=enterprise_name, created_at=datetime.utcnow())
        db.session.add(enterprise)
        db.session.flush()  # Get the enterprise ID before commit
        
        # Create admin user
        admin_user = AdminUser(
            username=username,
            password=hashed_password,
            enterprise_id=enterprise.id
        )
        db.session.add(admin_user)
        
        # Update plan status and associate with enterprise
        plan.payment_status = 'completed'
        plan.enterprise_id = enterprise.id
        
        # Create SQL Server user and database
        try:
            sqlserver_db_name, sqlserver_password = create_sqlserver_user_and_db(username, password)
            admin_user.sqlserver_password = sqlserver_password
        except Exception as e:
            logger.warning(f"Failed to create SQL Server user/database: {str(e)}")
        
        # Log the enterprise creation
        log_admin_action(
            username=username,
            action="Enterprise Created",
            action_description=f"Created enterprise {enterprise_name}",
            enterprise_id=enterprise.id
        )
        
        db.session.commit()
        
        flash('Enterprise created successfully! You can now setup database connection.', 'success')
        return redirect(url_for('verify_secret'))
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating enterprise: {str(e)}")
        flash(f'Error creating enterprise: {str(e)}', 'error')
        return redirect(url_for('enterprise'))
    
@app.route('/verify_enterprise', methods=['POST'])
def verify_enterprise():
    if request.method == 'POST':
        enterprise_name = request.form['enterprise_name']
        username = request.form['username']
        password = request.form['password']
        
        enterprise = Enterprise.query.filter_by(name=enterprise_name).first()
        if not enterprise:
            flash('Enterprise not found', 'error')
            return redirect(url_for('verify_secret'))
            
        # Verify admin user
        admin_user = AdminUser.query.filter_by(
            username=username,
            enterprise_id=enterprise.id
        ).first()
        
        if admin_user and bcrypt.checkpw(password.encode('utf-8'), admin_user.password.encode('utf-8')):
            session['enterprise_id'] = enterprise.id
            session['enterprise_name'] = enterprise.name
            session['username'] = username
            
            return redirect(url_for('admin_console'))
        
        flash('Invalid credentials', 'error')
        return redirect(url_for('verify_secret'))
    

@app.route('/verify_secret', methods=['GET', 'POST'])
def verify_secret():
    if request.method == 'POST':
        return verify_enterprise()
    return render_template('verify_secret.html')

@app.route('/admin_console')
@require_enterprise_auth
def admin_console():
    enterprise = get_current_enterprise()
    if not enterprise:
        flash("Enterprise not found", 'error')
        return redirect(url_for('verify_secret'))
        
    users = User.query.filter_by(enterprise_id=enterprise.id).all()
    return render_template('adminconsole.html',
                         users=users,
                         enterprise=enterprise)

@app.route('/audit_page')
@require_enterprise_auth
def audit_page():
    enterprise = get_current_enterprise()
    if not enterprise:
        flash("Enterprise not found", 'error')
        return redirect(url_for('verify_secret'))
        
    # Get logs for current enterprise, ordered by timestamp
    logs = AdminLog.query.filter_by(enterprise_id=enterprise.id)\
               .order_by(AdminLog.timestamp.desc()).all()
               
    return render_template('audit_page.html', 
                         logs=logs,
                         enterprise=enterprise)
    

def grant_sqlserver_database_access(username, password, database_names):
    try:
        enterprise = get_current_enterprise()
        if not enterprise or not enterprise.sqlserver_host:
            raise ValueError("SQL Server configuration not found")

        # Create server string based on whether instance name is provided
        server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
        
        # Create connection string for master database
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"UID={enterprise.sqlserver_admin_user};"
            f"PWD={enterprise.sqlserver_admin_password};"
            f"DATABASE=master;"
            f"Connection Timeout=30;"
        )
        
        conn = pyodbc.connect(conn_str)
        conn.autocommit = True
        cursor = conn.cursor()
        
        try:
            for db_name in database_names:
                # Skip personal database
                if db_name == sanitize_db_name(username):
                    continue
                    
                # Connect to the target database
                db_conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"UID={enterprise.sqlserver_admin_user};"
                    f"PWD={enterprise.sqlserver_admin_password};"
                    f"DATABASE={db_name};"
                )
                db_conn = pyodbc.connect(db_conn_str)
                db_cursor = db_conn.cursor()
                
                try:
                    # Create user in the database if not exists
                    db_cursor.execute(
                        f"IF NOT EXISTS (SELECT * FROM sys.database_principals WHERE name = ?) "
                        f"CREATE USER [{username}] FOR LOGIN [{username}]",
                        username
                    )
                    
                    # Grant SELECT permission on all tables
                    db_cursor.execute(f"""
                        DECLARE @sql NVARCHAR(MAX) = '';
                        SELECT @sql = @sql + 'GRANT SELECT ON ' + QUOTENAME(SCHEMA_NAME(schema_id)) + '.' + QUOTENAME(name) + ' TO [{username}]; '
                        FROM sys.tables;
                        EXEC sp_executesql @sql;
                    """)
                    
                    logger.info(f"Granted SELECT access to {username} on database {db_name}")
                finally:
                    db_cursor.close()
                    db_conn.close()
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        logger.error(f"Error granting SQL Server database access to {username}: {str(e)}")
        raise ValueError(f"Failed to grant database access: {str(e)}")


@app.route('/signup', methods=['GET', 'POST'])
@require_enterprise_auth
def signup():
    enterprise = get_current_enterprise()
    if not enterprise:
        flash("Enterprise not found", 'error')
        return redirect(url_for('verify_secret'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        access_list = request.form.getlist('access')  # List of JSON strings
        
        if not username or not password:
            flash("Username and password are required", 'error')
            return redirect(url_for('signup'))
        
        existing_user = User.query.filter_by(username=username, enterprise_id=enterprise.id).first()
        if existing_user:
            flash(f"Username '{username}' already exists in this enterprise", 'error')
            return redirect(url_for('signup'))
        
        try:
            new_access_list = []
            
            # Process selected databases from form
            for db_json in access_list:
                if not db_json.strip():  # Skip empty strings
                    continue
                
                try:
                    db_info = json.loads(db_json)
                    if not isinstance(db_info, dict):
                        logger.error(f"Invalid database entry (not a dict): {db_json}")
                        continue
                        
                    # Ensure required fields exist
                    if 'name' not in db_info or 'type' not in db_info:
                        logger.error(f"Invalid database entry (missing fields): {db_json}")
                        continue
                        
                    # Verify the database is actually shared with this enterprise
                    shared_db = SharedDatabase.query.filter_by(
                        database_name=db_info['name'], 
                        enterprise_id=enterprise.id,
                        database_type=db_info['type']
                    ).first()
                    
                    if shared_db:
                        # Only add if not already in the list
                        if not any(db['name'] == db_info['name'] for db in new_access_list):
                            new_access_list.append({
                                'name': db_info['name'],
                                'type': db_info['type']
                            })
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in access_list: {db_json}, error: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing database entry: {db_json}, error: {str(e)}")
                    continue
            
            if not new_access_list:
                flash("At least one database must be selected", 'error')
                return redirect(url_for('signup'))
            
            # Create user in SQLite database
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            new_user = User(
                username=username,
                password=hashed_password,
                access=json.dumps(new_access_list),
                enterprise_id=enterprise.id
            )
            db.session.add(new_user)
            
            # Create database users for each supported database type
            if enterprise.mysql_host:
                try:
                    mysql_db_name, mysql_password = create_mysql_user_and_db(username, password)
                    new_user.mysql_password = mysql_password
                except Exception as e:
                    logger.warning(f"Failed to create MySQL user/database: {str(e)}")
            
            if enterprise.postgres_host:
                try:
                    postgres_db_name, postgres_password = create_postgres_user_and_db(username, password)
                    new_user.postgres_password = postgres_password
                except Exception as e:
                    logger.warning(f"Failed to create PostgreSQL user/database: {str(e)}")
            
            if enterprise.sqlserver_host:
                try:
                    sqlserver_db_name, sqlserver_password = create_sqlserver_user_and_db(username, password)
                    new_user.sqlserver_password = sqlserver_password
                    
                    # Grant access to selected SQL Server databases
                    sqlserver_dbs = [db for db in new_access_list if db['type'] == 'sqlserver']
                    if sqlserver_dbs:
                        grant_sqlserver_database_access(username, sqlserver_password, [db['name'] for db in sqlserver_dbs])
                except Exception as e:
                    logger.warning(f"Failed to create SQL Server user/database: {str(e)}")
            
            db.session.commit()
            
            # Log the action
            log_admin_action(
                session['username'],
                "User created",
                f"Created user {username} with access to {json.dumps(new_access_list)}",
                enterprise_id=enterprise.id
            )
            
            flash(f"User {username} created successfully!", 'success')
            return redirect(url_for('signup'))
        
        except Exception as e:
            db.session.rollback()
            flash(f"Error creating user: {str(e)}", 'error')
            return redirect(url_for('signup'))
    
    shared_databases = [
        {'name': db.database_name, 'type': db.database_type}
        for db in SharedDatabase.query.filter_by(enterprise_id=enterprise.id).all()
    ]
    return render_template('signup.html', databases=shared_databases, enterprise=enterprise)
    
    

@app.route('/configuration', methods=['GET', 'POST'])
@require_enterprise_auth
def configuration():
    enterprise = get_current_enterprise()
    if not enterprise:
        flash("Enterprise not found", 'error')
        return redirect(url_for('verify_secret'))
    
    if request.method == 'POST':
        if 'form_type' in request.form and request.form['form_type'] == 'shared_databases':
            shared_databases = request.form.getlist('shared_databases')
            logger.debug(f"Received shared_databases: {shared_databases}")
            
            if not shared_databases:
                logger.warning("No shared databases submitted in the form")
                flash("No databases selected. Existing shared databases remain unchanged.", 'warning')
                return redirect(url_for('configuration'))
            
            try:
                parsed_databases = []
                for db_entry in shared_databases:
                    if not db_entry:
                        logger.warning("Empty database entry received, skipping")
                        continue
                    try:
                        db_info = json.loads(db_entry) if isinstance(db_entry, str) else db_entry
                        if not isinstance(db_info, dict) or 'name' not in db_info or 'type' not in db_info:
                            logger.error(f"Invalid database entry: {db_info}")
                            continue
                        if db_info['type'] not in ['mysql', 'postgres', 'sqlserver']:
                            logger.error(f"Invalid database type in entry: {db_info['type']}")
                            continue
                        parsed_databases.append(db_info)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in shared_databases: {db_entry}, error: {str(e)}")
                        continue
                
                logger.debug(f"Parsed databases: {parsed_databases}")
                if not parsed_databases:
                    flash("No valid databases provided. Existing shared databases remain unchanged.", 'warning')
                    return redirect(url_for('configuration'))
                
                if save_shared_databases(parsed_databases):
                    flash("Shared databases saved successfully!", 'success')
                    log_admin_action(
                        session['username'],
                        "Shared databases updated",
                        f"Updated shared databases: {json.dumps(parsed_databases)}",
                        enterprise_id=enterprise.id
                    )
                else:
                    flash("Error saving shared databases. Check logs for details.", 'error')
            except Exception as e:
                logger.error(f"Error processing shared databases: {str(e)}")
                flash(f"Error processing shared databases: {str(e)}", 'error')
            return redirect(url_for('configuration'))

    databases = []
    try:
        # MySQL databases
        if enterprise.mysql_host and enterprise.mysql_admin_user and enterprise.mysql_admin_password:
            engine = create_engine(
                f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}"
                f"@{enterprise.mysql_host}:{enterprise.mysql_port}"
            )
            with engine.connect() as conn:
                result = conn.execute(text("SHOW DATABASES"))
                system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys', 'admin'}
                databases.extend([{
                    'name': row[0],
                    'type': 'mysql',
                    'size': get_database_size(conn, row[0], 'mysql')
                } for row in result if row[0] not in system_dbs])
        
        # PostgreSQL databases
        if enterprise.postgres_host and enterprise.postgres_admin_user and enterprise.postgres_admin_password:
            engine = create_engine(
                f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}"
                f"@{enterprise.postgres_host}:{enterprise.postgres_port}/postgres"
            )
            with engine.connect() as conn:
                result = conn.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false"))
                system_dbs = {'postgres', 'template0', 'template1'}
                databases.extend([{
                    'name': row[0],
                    'type': 'postgres',
                    'size': get_database_size(conn, row[0], 'postgres')
                } for row in result if row[0] not in system_dbs])
                
        # SQL Server databases
        if enterprise.sqlserver_host and enterprise.sqlserver_admin_user and enterprise.sqlserver_admin_password:
            # Handle both instance and port-based connections
            if enterprise.sqlserver_instance:
                server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}"
            else:
                server = f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}" if enterprise.sqlserver_port else enterprise.sqlserver_host
                
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"UID={enterprise.sqlserver_admin_user};"
                f"PWD={enterprise.sqlserver_admin_password};"
                f"DATABASE=master;"
                f"Connection Timeout=30;"
            )
            
            try:
                conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                
                # Get all non-system databases
                cursor.execute("SELECT name FROM sys.databases WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb')")
                sqlserver_dbs = [row.name for row in cursor]
                
                for db_name in sqlserver_dbs:
                    size = get_database_size(cursor, db_name, 'sqlserver')
                    databases.append({
                        'name': db_name,
                        'type': 'sqlserver',
                        'size': size if size is not None else 'unknown'
                    })
                    
            except Exception as e:
                logger.error(f"Error fetching SQL Server databases: {str(e)}")
                flash(f"Error fetching SQL Server databases: {str(e)}", 'error')
            finally:
                if 'cursor' in locals():
                    cursor.close()
                if 'conn' in locals():
                    conn.close()
                
    except Exception as e:
        logger.error(f"Error fetching databases: {str(e)}")
        flash(f"Error fetching databases: {str(e)}", 'error')

    shared_databases = [
        {'name': db.database_name, 'type': db.database_type}
        for db in enterprise.shared_databases
    ]
    
    return render_template('configuration.html',
                         databases=databases,
                         shared_databases=shared_databases,
                         enterprise=enterprise)
                
def get_database_size(conn, db_name, db_type):
    try:
        if db_type == 'mysql':
            result = conn.execute(text(f"SELECT SUM(data_length + index_length) / 1024 / 1024 AS size_mb "
                                     f"FROM information_schema.TABLES "
                                     f"WHERE table_schema = '{db_name}'"))
            size = round(result.fetchone()[0] or 0, 2)
        elif db_type == 'postgres':
            result = conn.execute(text(f"SELECT pg_database_size('{db_name}') / 1024 / 1024"))
            size = round(result.fetchone()[0] / (1024 * 1024), 2)
        elif db_type == 'sqlserver':
            if isinstance(conn, pyodbc.Cursor):
                cursor = conn
                cursor.execute(
                    f"SELECT SUM(size) * 8.0 / 1024 FROM sys.master_files "
                    f"WHERE database_id = DB_ID('{db_name}')"
                )
                size = round(cursor.fetchone()[0] or 0, 2)
            else:
                cursor = conn.cursor()
                cursor.execute(
                    f"SELECT SUM(size) * 8.0 / 1024 FROM sys.master_files "
                    f"WHERE database_id = DB_ID('{db_name}')"
                )
                size = round(cursor.fetchone()[0] or 0, 2)
                cursor.close()
            return size
        else:
            return None
    except Exception:
        return None
    
def save_shared_databases(databases):
    try:
        if not hasattr(session, 'get'):
            raise RuntimeError("Flask session is not available. Ensure this function is called within a request context.")

        enterprise_id = session.get('enterprise_id')
        if not enterprise_id:
            raise ValueError("No enterprise session found")
            
        enterprise = db.session.get(Enterprise, enterprise_id)
        if not enterprise:
            raise ValueError("Enterprise not found")
            
        logger.debug(f"Enterprise SQL Server config: host={enterprise.sqlserver_host}, "
             f"instance={enterprise.sqlserver_instance}, port={enterprise.sqlserver_port}, "
             f"user={enterprise.sqlserver_admin_user}, password={'*' * (len(enterprise.sqlserver_admin_password) if enterprise.sqlserver_admin_password else 0)}")
        
        users = User.query.filter_by(enterprise_id=enterprise_id).all()
        logger.debug(f"Users for enterprise {enterprise_id}: {[user.username for user in users]}")
        
        mysql_engine = None
        postgres_engine = None
        sqlserver_conn = None
        
        # Initialize database connections
        try:
            if enterprise.mysql_host and enterprise.mysql_admin_user and enterprise.mysql_admin_password:
                mysql_engine = create_engine(
                    f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}"
                    f"@{enterprise.mysql_host}:{enterprise.mysql_port}"
                )
                logger.debug("MySQL connection initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MySQL connection: {str(e)}")
            mysql_engine = None
            
        try:
            if enterprise.postgres_host and enterprise.postgres_admin_user and enterprise.postgres_admin_password:
                postgres_engine = create_engine(
                    f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}"
                    f"@{enterprise.postgres_host}:{enterprise.postgres_port}/postgres",
                    isolation_level="AUTOCOMMIT"
                )
                logger.debug("PostgreSQL connection initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize PostgreSQL connection: {str(e)}")
            postgres_engine = None
            
        try:
            if enterprise.sqlserver_host and enterprise.sqlserver_admin_user and enterprise.sqlserver_admin_password:
                server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"UID={enterprise.sqlserver_admin_user};"
                    f"PWD={enterprise.sqlserver_admin_password};"
                    f"DATABASE=master"
                )
                logger.debug(f"SQL Server connection string: {conn_str}")
                sqlserver_conn = pyodbc.connect(conn_str)
                # Create SQL Server logins for users if they don't exist
                cursor = sqlserver_conn.cursor()
                for user in users:
                    escaped_username = user.username.replace("'", "''")
                    cursor.execute(
                        f"IF NOT EXISTS (SELECT * FROM sys.server_principals WHERE name = '{escaped_username}') "
                        f"CREATE LOGIN [{escaped_username}] WITH PASSWORD = '{user.sqlserver_password or 'DefaultSecurePassword123!'}'"
                    )
                sqlserver_conn.commit()
                cursor.close()
                logger.debug("SQL Server connection initialized and logins verified")
        except Exception as e:
            logger.warning(f"Failed to initialize SQL Server connection: {str(e)}")
            flash(f"SQL Server connection failed: {str(e)}. SQL Server databases may not be updated.", 'warning')
            sqlserver_conn = None
        
        current_shared_dbs = {
            (db.database_name, db.database_type, db.enterprise_id): db
            for db in SharedDatabase.query.filter_by(enterprise_id=enterprise_id).all()
        }
        logger.debug(f"Current shared databases: {list(current_shared_dbs.keys())}")
        
        new_databases = set()
        for db_entry in databases:
            if isinstance(db_entry, str):
                try:
                    db_info = json.loads(db_entry)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in shared_databases: {db_entry}, error: {str(e)}")
                    continue
            elif isinstance(db_entry, dict):
                db_info = db_entry
            else:
                logger.error(f"Invalid database entry type: {type(db_entry)}")
                continue

            db_name = db_info.get('name')
            db_type = db_info.get('type')
            if not db_name or not db_type:
                logger.error(f"Missing name or type in database entry: {db_info}")
                continue
            if db_type not in ['mysql', 'postgres', 'sqlserver']:
                logger.error(f"Invalid database type: {db_type}")
                continue
            new_databases.add((db_name, db_type, enterprise_id))
        
        logger.debug(f"New databases to share: {new_databases}")

        for (db_name, db_type, db_enterprise_id), shared_db in list(current_shared_dbs.items()):
            if (db_name, db_type, db_enterprise_id) not in new_databases:
                for user in users:
                    escaped_username = user.username.replace("'", "''")
                    escaped_db_name = db_name.replace("'", "''")
                    if db_type == 'mysql' and mysql_engine:
                        try:
                            with mysql_engine.connect() as conn:
                                result = conn.execute(text("SHOW DATABASES"))
                                mysql_dbs = [row[0] for row in result]
                                if db_name in mysql_dbs:
                                    conn.execute(
                                        text(f"REVOKE SELECT ON `{escaped_db_name}`.* FROM '{escaped_username}'@'%'")
                                    )
                                    conn.execute(
                                        text(f"REVOKE SELECT ON `{escaped_db_name}`.* FROM '{escaped_username}'@'localhost'")
                                    )
                                    conn.execute(text("FLUSH PRIVILEGES"))
                                    logger.info(f"Revoked SELECT privileges from {user.username} on MySQL database {db_name}")
                        except Exception as e:
                            logger.warning(f"Error revoking MySQL SELECT privileges from {user.username} on {db_name}: {str(e)}")
                    elif db_type == 'postgres' and postgres_engine:
                        try:
                            with postgres_engine.connect() as conn:
                                result = conn.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false"))
                                postgres_dbs = [row[0] for row in result]
                                if db_name in postgres_dbs:
                                    with create_engine(
                                        f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}"
                                        f"@{enterprise.postgres_host}:{enterprise.postgres_port}/{db_name}",
                                        isolation_level="AUTOCOMMIT"
                                    ).connect() as db_conn:
                                        db_conn.execute(
                                            text(f"REVOKE SELECT ON ALL TABLES IN SCHEMA public FROM {escaped_username}")
                                        )
                                        logger.info(f"Revoked SELECT privileges from {user.username} on PostgreSQL database {db_name}")
                        except Exception as e:
                            logger.warning(f"Error revoking PostgreSQL SELECT privileges from {user.username} on {db_name}: {str(e)}")
                    elif db_type == 'sqlserver' and sqlserver_conn:
                        try:
                            cursor = sqlserver_conn.cursor()
                            cursor.execute(
                                f"SELECT name FROM sys.databases WHERE name = '{escaped_db_name}'"
                            )
                            if cursor.fetchone():
                                cursor.execute(
                                    f"USE [{escaped_db_name}]; "
                                    f"IF EXISTS (SELECT * FROM sys.database_principals WHERE name = '{escaped_username}') "
                                    f"REVOKE SELECT TO [{escaped_username}];"
                                )
                                sqlserver_conn.commit()
                                logger.info(f"Revoked SELECT permissions from {user.username} on SQL Server database {db_name}")
                            cursor.close()
                        except Exception as e:
                            logger.warning(f"Error revoking SQL Server SELECT permissions from {user.username} on {db_name}: {str(e)}")
                db.session.delete(shared_db)
                logger.info(f"Removed shared database {db_name} ({db_type}) for enterprise {enterprise_id}")

        for db_name, db_type, db_enterprise_id in new_databases:
            if (db_name, db_type, db_enterprise_id) not in current_shared_dbs:
                shared_db = SharedDatabase(
                    database_name=db_name,
                    database_type=db_type,
                    enterprise_id=enterprise_id
                )
                db.session.add(shared_db)
                
                for user in users:
                    escaped_username = user.username.replace("'", "''")
                    escaped_db_name = db_name.replace("'", "''")
                    if db_type == 'mysql' and mysql_engine:
                        try:
                            with mysql_engine.connect() as conn:
                                result = conn.execute(text("SHOW DATABASES"))
                                mysql_dbs = [row[0] for row in result]
                                if db_name in mysql_dbs:
                                    conn.execute(
                                        text(f"GRANT SELECT ON `{escaped_db_name}`.* TO '{escaped_username}'@'%'")
                                    )
                                    conn.execute(
                                        text(f"GRANT SELECT ON `{escaped_db_name}`.* TO '{escaped_username}'@'localhost'")
                                    )
                                    conn.execute(text("FLUSH PRIVILEGES"))
                                    logger.info(f"Granted SELECT privileges to {user.username} on MySQL shared database {db_name}")
                        except Exception as e:
                            logger.warning(f"Error granting MySQL SELECT privileges to {user.username} on {db_name}: {str(e)}")
                    elif db_type == 'postgres' and postgres_engine:
                        try:
                            with postgres_engine.connect() as conn:
                                result = conn.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false"))
                                postgres_dbs = [row[0] for row in result]
                                if db_name in postgres_dbs:
                                    with create_engine(
                                        f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}"
                                        f"@{enterprise.postgres_host}:{enterprise.postgres_port}/{db_name}",
                                        isolation_level="AUTOCOMMIT"
                                    ).connect() as db_conn:
                                        db_conn.execute(
                                            text(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {escaped_username}")
                                        )
                                        logger.info(f"Granted SELECT privileges to {user.username} on PostgreSQL shared database {db_name}")
                        except Exception as e:
                            logger.warning(f"Error granting PostgreSQL SELECT privileges to {user.username} on {db_name}: {str(e)}")
                    elif db_type == 'sqlserver' and sqlserver_conn:
                        try:
                            cursor = sqlserver_conn.cursor()
                            cursor.execute(
                                f"SELECT name FROM sys.databases WHERE name = '{escaped_db_name}'"
                            )
                            if cursor.fetchone():
                                cursor.execute(
                                    f"USE [{escaped_db_name}]; "
                                    f"IF NOT EXISTS (SELECT * FROM sys.database_principals WHERE name = '{escaped_username}') "
                                    f"CREATE USER [{escaped_username}] FOR LOGIN [{escaped_username}]; "
                                    f"GRANT SELECT TO [{escaped_username}];"
                                )
                                sqlserver_conn.commit()
                                logger.info(f"Granted SELECT permissions to {user.username} on SQL Server shared database {db_name}")
                            else:
                                logger.warning(f"SQL Server database {db_name} not found, skipping permission grant for {user.username}")
                            cursor.close()
                        except Exception as e:
                            logger.warning(f"Error granting SQL Server SELECT permissions to {user.username} on {db_name}: {str(e)}")

        db.session.commit()
        if sqlserver_conn:
            try:
                sqlserver_conn.close()
            except Exception as e:
                logger.warning(f"Error closing SQL Server connection: {str(e)}")
        logger.info(f"Shared databases updated successfully for enterprise {enterprise_id}")
        return True
    except Exception as e:
        db.session.rollback()
        if 'sqlserver_conn' in locals() and sqlserver_conn:
            try:
                sqlserver_conn.close()
            except Exception as e:
                logger.warning(f"Error closing SQL Server connection: {str(e)}")
        logger.error(f"Error saving shared databases: {str(e)}")
        flash(f"Error saving shared databases: {str(e)}", 'error')
        return False

    
@app.route('/api/save-mysql-config', methods=['POST'])
@require_enterprise_auth
def api_save_mysql_config():
    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'success': False, 'error': 'Enterprise not found'}), 404

        data = request.get_json()
        
        # Validate required fields
        required_fields = ['host', 'port', 'user', 'password']
        if not all(field in data for field in required_fields):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Test the connection
        connection_string = f"mysql+pymysql://{data['user']}:{data['password']}@{data['host']}:{data['port']}"
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {str(e)}")
            return jsonify({'success': False, 'error': f'Connection test failed: {str(e)}'}), 400

        # Update enterprise with MySQL config
        enterprise.mysql_host = data['host']
        enterprise.mysql_port = data['port']
        enterprise.mysql_admin_user = data['user']
        enterprise.mysql_admin_password = data['password']
        db.session.commit()

        # Save to file for backward compatibility
        with open(MYSQL_CONFIG_FILE, 'w') as f:
            json.dump(data, f)
        
        # Update app config
        app.config['MYSQL_HOST'] = data['host']
        app.config['MYSQL_PORT'] = data['port']
        app.config['MYSQL_ADMIN_USER'] = data['user']
        app.config['MYSQL_ADMIN_PASSWORD'] = data['password']
        
        # Get all non-system databases with sizes
        databases = []
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SHOW DATABASES"))
                system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys', 'admin'}
                for row in result:
                    db_name = row[0]
                    if db_name not in system_dbs:
                        size = get_database_size(conn, db_name, 'mysql')
                        databases.append({
                            'name': db_name,
                            'type': 'mysql',
                            'size': size if size is not None else 'unknown'
                        })
            logger.info(f"Fetched MySQL databases: {databases}")
        except Exception as e:
            logger.error(f"Error fetching MySQL databases: {str(e)}")
            return jsonify({'success': False, 'error': f'Failed to fetch databases: {str(e)}'}), 500

        # Get current shared databases
        shared_dbs = [
            {'name': db.database_name, 'type': db.database_type}
            for db in SharedDatabase.query.filter_by(enterprise_id=enterprise.id).all()
        ]
        
        return jsonify({
            'success': True,
            'databases': databases,
            'shared_databases': shared_dbs,
            'message': 'MySQL configuration saved successfully'
        })
            
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving MySQL config: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error saving configuration: {str(e)}'
        }), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            session['username'] = username
            session['enterprise_id'] = user.enterprise_id
            session['mysql_password'] = user.mysql_password
            session['postgres_password'] = user.postgres_password
            session['sqlserver_password'] = user.sqlserver_password
            log_admin_action(username, "Logged in", enterprise_id=user.enterprise_id)
            return redirect(url_for('app_page'))
        else:
            return render_template('login.html', message="Invalid username or password")
    return render_template('login.html')

@app.route('/app')
def app_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('app.html')

@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if 'username' not in session:
        flash("Please log in to change your password.", 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = session['username']
        new_password = request.form['new_password']
        user = User.query.filter_by(username=username).first()

        if user:
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            user.password = hashed_password.decode('utf-8')

            enterprise = get_current_enterprise()
            if not enterprise:
                flash("Enterprise not found", 'error')
                return redirect(url_for('change_password'))

            # Update MySQL password if configured
            if enterprise.mysql_host and enterprise.mysql_admin_user and enterprise.mysql_admin_password:
                try:
                    admin_engine = create_engine(
                        f"mysql+pymysql://{app.config['MYSQL_ADMIN_USER']}:{app.config['MYSQL_ADMIN_PASSWORD']}@{app.config['MYSQL_HOST']}"
                    )
                    with admin_engine.connect() as conn:
                        escaped_username = username.replace("'", "''")
                        escaped_password = new_password.replace("'", "''")
                        conn.execute(
                            text(f"ALTER USER '{escaped_username}'@'%' IDENTIFIED BY '{escaped_password}'")
                        )
                        conn.execute(
                            text(f"ALTER USER '{escaped_username}'@'localhost' IDENTIFIED BY '{escaped_password}'")
                        )
                        conn.execute(text("FLUSH PRIVILEGES"))
                        user.mysql_password = new_password
                        logger.info(f"Updated MySQL user password for {username}")
                except Exception as e:
                    logger.error(f"Error updating MySQL user password: {str(e)}")
                    flash(f"Error updating MySQL password: {str(e)}", 'error')

            # Update PostgreSQL password if configured
            if enterprise.postgres_host and enterprise.postgres_admin_user and enterprise.postgres_admin_password:
                try:
                    admin_engine = create_engine(
                        f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}"
                        f"@{enterprise.postgres_host}:{enterprise.postgres_port}/postgres",
                        isolation_level="AUTOCOMMIT"
                    )
                    with admin_engine.connect() as conn:
                        escaped_username = username.replace("'", "''")
                        escaped_password = new_password.replace("'", "''")
                        conn.execute(
                            text(f"ALTER USER {escaped_username} WITH PASSWORD '{escaped_password}'")
                        )
                        user.postgres_password = new_password
                        logger.info(f"Updated PostgreSQL user password for {username}")
                except Exception as e:
                    logger.error(f"Error updating PostgreSQL user password: {str(e)}")
                    flash(f"Error updating PostgreSQL password: {str(e)}", 'error')

            # Update SQL Server password if configured
            if enterprise.sqlserver_host and enterprise.sqlserver_admin_user and enterprise.sqlserver_admin_password:
                try:
                    # Create server string based on whether instance name is provided
                    server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
                    conn_str = (
                        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                        f"SERVER={server};"
                        f"UID={enterprise.sqlserver_admin_user};"
                        f"PWD={enterprise.sqlserver_admin_password};"
                        f"DATABASE=master"
                    )
                    conn = pyodbc.connect(conn_str)
                    conn.autocommit = True
                    cursor = conn.cursor()
                    escaped_username = username.replace("'", "''")
                    escaped_password = new_password.replace("'", "''")
                    cursor.execute(
                        f"ALTER LOGIN [{escaped_username}] WITH PASSWORD = '{escaped_password}'"
                    )
                    user.sqlserver_password = new_password
                    cursor.close()
                    conn.close()
                    logger.info(f"Updated SQL Server user password for {username}")
                except Exception as e:
                    logger.error(f"Error updating SQL Server user password: {str(e)}")
                    flash(f"Error updating SQL Server password: {str(e)}", 'error')

            db.session.commit()
            log_admin_action(username, "Changed password")
            
            flash("Password changed successfully!", 'success')
            return redirect(url_for('change_password'))
        else:
            flash("User not found.", 'error')
            return redirect(url_for('change_password'))

    return render_template('changepassword.html', username=session['username'])



def sanitize_db_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())

@app.template_filter('from_json')
def from_json_filter(data):
    try:
        return json.loads(data) if data else []
    except json.JSONDecodeError as e:
        app.logger.error(f"JSON decode error in from_json filter: {str(e)}")
        return []

@app.route('/admin_update', methods=['GET', 'POST'])
def admin_update():
    from database import db
    
    shared_databases = load_shared_databases()
    selected_username = request.args.get('username') or (request.form.get('username') if request.method == 'POST' else None)

    if request.method == 'POST':
        username = request.form.get('username')
        update_option = request.form.get('update_option')
        app.logger.info(f"Processing update for username: {username}, update_option: {update_option}")

        if not username:
            flash("No username provided.", 'error')
            return redirect(url_for('admin_update'))

        user = User.query.filter_by(username=username).first()
        if not user:
            flash(f'User with username {username} not found.', 'error')
            return redirect(url_for('admin_update'))

        enterprise = get_current_enterprise()
        if not enterprise:
            flash("Enterprise not found", 'error')
            return redirect(url_for('admin_update'))

        updated = False

        if update_option in ['password', 'both']:
            new_password = request.form.get('new_password', '').strip()
            app.logger.info(f"New password provided: {'Yes' if new_password else 'No'}")
            if new_password:
                hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
                user.password = hashed_password.decode('utf-8')
                
                # Update MySQL password if configured
                if enterprise.mysql_host and enterprise.mysql_admin_user and enterprise.mysql_admin_password:
                    try:
                        admin_engine = create_engine(
                            f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}@{enterprise.mysql_host}:{enterprise.mysql_port}/mysql"
                        )
                        with admin_engine.connect() as conn:
                            escaped_username = username.replace("'", "''")
                            escaped_password = new_password.replace("'", "''")
                            conn.execute(
                                text(f"ALTER USER '{escaped_username}'@'%' IDENTIFIED BY '{escaped_password}'")
                            )
                            conn.execute(
                                text(f"ALTER USER '{escaped_username}'@'localhost' IDENTIFIED BY '{escaped_password}'")
                            )
                            conn.execute(text("FLUSH PRIVILEGES"))
                            user.mysql_password = new_password
                            app.logger.info(f"Updated MySQL user password for {username}")
                    except Exception as e:
                        app.logger.error(f"Error updating MySQL user password: {str(e)}")
                        flash(f"Error updating MySQL password: {str(e)}", 'error')

                # Update PostgreSQL password if configured
                if enterprise.postgres_host and enterprise.postgres_admin_user and enterprise.postgres_admin_password:
                    try:
                        admin_engine = create_engine(
                            f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}"
                            f"@{enterprise.postgres_host}:{enterprise.postgres_port}/postgres",
                            isolation_level="AUTOCOMMIT"
                        )
                        with admin_engine.connect() as conn:
                            escaped_username = username.replace("'", "''")
                            escaped_password = new_password.replace("'", "''")
                            conn.execute(
                                text(f"ALTER USER {escaped_username} WITH PASSWORD '{escaped_password}'")
                            )
                            user.postgres_password = new_password
                            app.logger.info(f"Updated PostgreSQL user password for {username}")
                    except Exception as e:
                        app.logger.error(f"Error updating PostgreSQL user password: {str(e)}")
                        flash(f"Error updating PostgreSQL password: {str(e)}", 'error')
                        
                # Update SQL Server password if configured
                if enterprise.sqlserver_host and enterprise.sqlserver_admin_user and enterprise.sqlserver_admin_password:
                    try:
                        server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
                        conn_str = (
                            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                            f"SERVER={server};"
                            f"UID={enterprise.sqlserver_admin_user};"
                            f"PWD={enterprise.sqlserver_admin_password};"
                            f"DATABASE=master"
                        )
                        conn = pyodbc.connect(conn_str)
                        conn.autocommit = True
                        cursor = conn.cursor()
                        escaped_username = username.replace("'", "''")
                        escaped_password = new_password.replace("'", "''")
                        cursor.execute(
                            f"ALTER LOGIN [{escaped_username}] WITH PASSWORD = '{escaped_password}'"
                        )
                        user.sqlserver_password = new_password
                        cursor.close()
                        conn.close()
                        app.logger.info(f"Updated SQL Server user password for {username}")
                    except Exception as e:
                        app.logger.error(f"Error updating SQL Server user password: {str(e)}")
                        flash(f"Error updating SQL Server password: {str(e)}", 'error')
                
                updated = True

        if update_option in ['access', 'both']:
            access_list = request.form.getlist('access')
            personal_db = sanitize_db_name(username)
            new_access_list = []
            for db_json in access_list:
                try:
                    db_info = json.loads(db_json) if isinstance(db_json, str) else db_json
                    if not isinstance(db_info, dict) or 'name' not in db_info or 'type' not in db_info:
                        app.logger.error(f"Invalid database entry: {db_json}")
                        continue
                    new_access_list.append(db_info)
                except json.JSONDecodeError as e:
                    app.logger.error(f"Invalid JSON in access_list: {db_json}, error: {str(e)}")
                    continue
            
            # Ensure personal database is included
            if personal_db not in [db['name'] for db in new_access_list]:
                new_access_list.append({'name': personal_db, 'type': 'mysql'})
            
            current_access = json.loads(user.access) if user.access else []
            if new_access_list != current_access:
                user.access = json.dumps(new_access_list)
                updated = True
            else:
                app.logger.info(f"No changes in access for {username}")

        if updated:
            try:
                db.session.commit()
                flash(f"User credentials updated successfully for {username}.", 'success')
            except Exception as e:
                db.session.rollback()
                app.logger.error(f"Error committing changes for user {username}: {str(e)}")
                flash(f"Error saving changes: {str(e)}", 'error')
        elif update_option:
            flash(f"No changes were made for {username}.", 'info')

        return redirect(url_for('admin_update', username=username))

    # GET request handling
    enterprise = get_current_enterprise()
    if not enterprise:
        flash("Enterprise not found", 'error')
        return redirect(url_for('admin_update'))

    databases = []
    if selected_username:
        try:
            # Get all available databases (using admin credentials)
            admin_engine_mysql = None
            admin_engine_postgres = None
            all_databases = []
            
            # Get SQL Server databases
            if enterprise.sqlserver_host and enterprise.sqlserver_admin_user and enterprise.sqlserver_admin_password:
                server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"UID={enterprise.sqlserver_admin_user};"
                    f"PWD={enterprise.sqlserver_admin_password};"
                    f"DATABASE=master"
                )
                conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sys.databases WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb')")
                all_databases.extend([{'name': row.name, 'type': 'sqlserver'} for row in cursor])
                cursor.close()
                conn.close()
            
            if enterprise.mysql_host and enterprise.mysql_admin_user and enterprise.mysql_admin_password:
                admin_engine_mysql = create_engine(
                    f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}@{enterprise.mysql_host}:{enterprise.mysql_port}/mysql"
                )
                with admin_engine_mysql.connect() as connection:
                    result = connection.execute(text("SHOW DATABASES"))
                    system_dbs = {'information_schema', 'mysql', 'performance_schema', 'sys', 'admin'}
                    all_databases.extend([{'name': row[0], 'type': 'mysql'} for row in result if row[0] not in system_dbs])
            
            if enterprise.postgres_host and enterprise.postgres_admin_user and enterprise.postgres_admin_password:
                admin_engine_postgres = create_engine(
                    f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}"
                    f"@{enterprise.postgres_host}:{enterprise.postgres_port}/postgres"
                )
                with admin_engine_postgres.connect() as connection:
                    result = connection.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false"))
                    system_dbs = {'postgres', 'template0', 'template1'}
                    all_databases.extend([{'name': row[0], 'type': 'postgres'} for row in result if row[0] not in system_dbs])
                    
            # Get user's current access
            user = User.query.filter_by(username=selected_username).first()
            if user and user.access:
                databases = json.loads(user.access) if user.access else []
                # Add any shared databases that aren't already in the list
                for db in shared_databases:
                    if {'name': db, 'type': 'mysql'} not in databases and {'name': db, 'type': 'postgres'} not in databases:
                        databases.append({'name': db, 'type': 'mysql'})  # Adjust type as needed
            else:
                databases = [{'name': db, 'type': 'mysql'} for db in shared_databases]
        except Exception as e:
            app.logger.error(f"Error fetching databases: {str(e)}")
            flash(f"Error fetching databases: {str(e)}", 'error')

    users = User.query.all()
    user = User.query.filter_by(username=selected_username).first() if selected_username else None
    
    return render_template('adminupdate.html', 
                         users=users, 
                         databases=databases,
                         selected_username=selected_username,
                         user=user)
    
@app.route('/delete_user', methods=['POST'])
def delete_user():
    if request.method == 'POST':
        username = request.form['username']
        source_page = request.form.get('source_page', 'admin_update')
        user = User.query.filter_by(username=username).first()

        if user:
            try:
                # Delete associated chat threads from database
                try:
                    from models import ChatThread
                    ChatThread.query.filter_by(user_id=user.id).delete()
                    logger.info(f"Deleted chat threads for user {username} from database")
                except Exception as e:
                    logger.warning(f"Error deleting chat threads from database for {username}: {str(e)}")

                # Delete associated chat thread file
                try:
                    chat_thread_path = os.path.join(app.config['UPLOAD_FOLDER'], 'chat_threads', f"{username}.json")
                    if os.path.exists(chat_thread_path):
                        os.remove(chat_thread_path)
                        logger.info(f"Deleted chat thread file for user {username}")
                except Exception as e:
                    logger.warning(f"Error deleting chat thread file for {username}: {str(e)}")

                # Get enterprise details
                enterprise = Enterprise.query.get(user.enterprise_id)
                if not enterprise:
                    raise ValueError("Enterprise not found")

                # Drop SQL Server database and login
                if enterprise.sqlserver_host and enterprise.sqlserver_admin_user and enterprise.sqlserver_admin_password:
                    try:
                        # Handle both instance and port-based connections
                        server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
                        conn_str = (
                            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                            f"SERVER={server};"
                            f"UID={enterprise.sqlserver_admin_user};"
                            f"PWD={enterprise.sqlserver_admin_password};"
                            f"DATABASE=master"
                        )
                        conn = pyodbc.connect(conn_str)
                        conn.autocommit = True
                        cursor = conn.cursor()
                        db_name = sanitize_db_name(username)
                        # Terminate active connections and drop database
                        cursor.execute(f"IF DB_ID('{db_name}') IS NOT NULL BEGIN ALTER DATABASE [{db_name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE; DROP DATABASE [{db_name}]; END")
                        # Drop login if it exists
                        cursor.execute(f"IF SUSER_ID('{username}') IS NOT NULL DROP LOGIN [{username}]")
                        cursor.close()
                        conn.close()
                        logger.info(f"Dropped SQL Server user {username} and database {db_name}")
                    except Exception as e:
                        logger.error(f"Error dropping SQL Server user/database: {str(e)}")
                        flash(f"Error dropping SQL Server user/database: {str(e)}", 'error')

                # Drop MySQL database and user
                if enterprise.mysql_host and enterprise.mysql_admin_user and enterprise.mysql_admin_password:
                    try:
                        mysql_engine = create_engine(
                            f"mysql+mysqlconnector://{enterprise.mysql_admin_user}:"
                            f"{enterprise.mysql_admin_password}@{enterprise.mysql_host}:"
                            f"{enterprise.mysql_port}/mysql"
                        )
                        with mysql_engine.connect() as conn:
                            db_name = sanitize_db_name(username)
                            conn.execute(text(f"DROP DATABASE IF EXISTS `{db_name}`"))
                            conn.execute(text(f"DROP USER IF EXISTS '{username}'@'%'"))
                        logger.info(f"Dropped MySQL user {username} and database {db_name}")
                    except Exception as e:
                        logger.error(f"Error dropping MySQL user/database: {str(e)}")
                        flash(f"Error dropping MySQL user/database: {str(e)}", 'error')

                # Drop PostgreSQL database and user
                if enterprise.postgres_host and enterprise.postgres_admin_user and enterprise.postgres_admin_password:
                    try:
                        postgres_engine = create_engine(
                            f"postgresql://{enterprise.postgres_admin_user}:"
                            f"{enterprise.postgres_admin_password}@{enterprise.postgres_host}:"
                            f"{enterprise.postgres_port}/postgres",
                            isolation_level="AUTOCOMMIT"
                        )
                        with postgres_engine.connect() as conn:
                            db_name = sanitize_db_name(username)
                            # Terminate any active connections to the database
                            conn.execute(text(
                                f"SELECT pg_terminate_backend(pg_stat_activity.pid) "
                                f"FROM pg_stat_activity "
                                f"WHERE pg_stat_activity.datname = '{db_name}' "
                                f"AND pid <> pg_backend_pid()"
                            ))
                            # Drop the database with FORCE to handle dependencies
                            conn.execute(text(f"DROP DATABASE IF EXISTS \"{db_name}\" WITH (FORCE)"))
                            # Revoke privileges and drop owned objects
                            conn.execute(text(f"REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM \"{username}\""))
                            conn.execute(text(f"REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM \"{username}\""))
                            conn.execute(text(f"REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public FROM \"{username}\""))
                            conn.execute(text(f"REVOKE ALL PRIVILEGES ON SCHEMA public FROM \"{username}\""))
                            conn.execute(text(f"DROP OWNED BY \"{username}\""))
                            # Drop the user role
                            conn.execute(text(f"DROP ROLE IF EXISTS \"{username}\""))
                        logger.info(f"Dropped PostgreSQL user {username} and database {db_name}")
                    except Exception as e:
                        logger.error(f"Error dropping PostgreSQL user/database: {str(e)}")
                        flash(f"Error dropping PostgreSQL user/database: {str(e)}", 'error')

                # Delete user-specific files
                try:
                    user_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'users', username)
                    if os.path.exists(user_dir):
                        import shutil
                        shutil.rmtree(user_dir)
                        logger.info(f"Deleted user directory for {username}")
                except Exception as e:
                    logger.warning(f"Error deleting user directory for {username}: {str(e)}")

                # Delete user from SQLite database
                db.session.delete(user)
                db.session.commit()
                flash(f'User "{username}" deleted successfully.', 'success')
                logger.info(f"User {username} deleted successfully by {session.get('username')}")
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error deleting user {username}: {str(e)}")
                flash(f'Error deleting user: {str(e)}', 'error')
        else:
            flash(f'User "{username}" does not exist.', 'error')
            logger.warning(f"Attempted to delete non-existent user {username}")

        return redirect(url_for(source_page))
    return redirect(url_for('admin_update'))

@app.route('/logout')
def logout():
    if 'username' in session:
        username = session['username']
        enterprise_id = session.get('enterprise_id')
        log_admin_action(username, "Logged out", enterprise_id=enterprise_id)
        # Clear all session data
        session.clear()
    return redirect(url_for('home'))

@app.route('/api/logout', methods=['POST'])
def api_logout():
    try:
        if 'username' in session:
            log_admin_action(session['username'], "Logged out via API")
            session.pop('username', None)
            session.pop('access', None)
            session.pop('mysql_password', None)
            session.pop('current_database', None)
            session.pop('connection_type', None)
            session.pop('current_vectorstore', None)
        return jsonify({'success': True, 'message': 'Logged out successfully'})
    except Exception as e:
        logger.error(f"Error during API logout: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user-databases', methods=['GET'])
def get_user_databases():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'error': 'Enterprise not found'}), 404

        databases = []
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        access_list = json.loads(user.access) if user.access else []

        # Get MySQL databases
        if enterprise.mysql_host:
            try:
                engine = create_engine(
                    f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}"
                    f"@{enterprise.mysql_host}:{enterprise.mysql_port}"
                )
                with engine.connect() as conn:
                    result = conn.execute(text("SHOW DATABASES"))
                    mysql_dbs = [row[0] for row in result]
                    user_db = sanitize_db_name(session['username'])
                    for db in mysql_dbs:
                        if db == user_db or any(acc['name'] == db and acc['type'] == 'mysql' for acc in access_list):
                            databases.append({'name': db, 'type': 'mysql'})
            except Exception as e:
                logger.error(f"Error fetching MySQL databases: {str(e)}")

        # Get PostgreSQL databases
        if enterprise.postgres_host:
            try:
                engine = create_engine(
                    f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}"
                    f"@{enterprise.postgres_host}:{enterprise.postgres_port}/postgres"
                )
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false"))
                    postgres_dbs = [row[0] for row in result]
                    user_db = sanitize_db_name(session['username'])
                    for db in postgres_dbs:
                        if db == user_db or any(acc['name'] == db and acc['type'] == 'postgres' for acc in access_list):
                            databases.append({'name': db, 'type': 'postgres'})
            except Exception as e:
                logger.error(f"Error fetching PostgreSQL databases: {str(e)}")

        # Get SQL Server databases
        if enterprise.sqlserver_host:
            try:
                server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"UID={enterprise.sqlserver_admin_user};"
                    f"PWD={enterprise.sqlserver_admin_password};"
                    f"DATABASE=master"
                )
                conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sys.databases WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb')")
                sqlserver_dbs = [row.name for row in cursor]
                user_db = sanitize_db_name(session['username'])
                for db in sqlserver_dbs:
                    if db == user_db or any(acc['name'] == db and acc['type'] == 'sqlserver' for acc in access_list):
                        databases.append({'name': db, 'type': 'sqlserver'})
                cursor.close()
                conn.close()
            except Exception as e:
                logger.error(f"Error fetching SQL Server databases: {str(e)}")

        return jsonify({'databases': databases})
    except Exception as e:
        logger.error(f"Error fetching user databases: {str(e)}")
        return jsonify({'error': str(e)}), 500

from flask import session  # Ensure Flask session is imported



@app.route('/api/current-user')
def current_user():
    username = session.get('username')
    if not username:
        return jsonify({'error': 'Not logged in'}), 401
    return jsonify({'username': username})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'})
    if file:
        try:
            vectordb = process_pdf(file)
            session['vectordb'] = vectordb
            return jsonify({'success': True, 'message': 'PDF processed and ready for Q&A!'})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error processing PDF: {e}'})

@app.route('/api/connect-database', methods=['POST'])
def connect_database():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.get_json()
    selected_db = data.get('database')
    db_type = data.get('type')  # Expect 'mysql', 'postgres', or 'sqlserver'

    if not selected_db or not db_type:
        return jsonify({'error': 'Database name and type are required'}), 400

    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'error': 'Enterprise not found'}), 404

        user = User.query.filter_by(username=session['username']).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        shared_databases = [
            {'name': db.database_name, 'type': db.database_type}
            for db in SharedDatabase.query.filter_by(enterprise_id=enterprise.id).all()
        ]
        is_shared = any(db['name'] == selected_db and db['type'] == db_type for db in shared_databases)

        db_uri = None
        if db_type == 'sqlserver':
            if not enterprise.sqlserver_host or not enterprise.sqlserver_admin_user or not enterprise.sqlserver_admin_password:
                return jsonify({'error': 'SQL Server configuration not found'}), 400
            
            # Create server string based on whether instance name is provided
            server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
            
            if is_shared:
                # Use admin credentials for shared databases
                db_uri = (
                    f"mssql+pyodbc://{enterprise.sqlserver_admin_user}:{enterprise.sqlserver_admin_password}"
                    f"@{server}/{selected_db}"
                    f"?driver=ODBC+Driver+17+for+SQL+Server"
                )
            else:
                # Use user credentials for personal databases
                if not user.sqlserver_password:
                    return jsonify({'error': 'SQL Server credentials not found for user'}), 401
                db_uri = (
                    f"mssql+pyodbc://{session['username']}:{user.sqlserver_password}"
                    f"@{server}/{selected_db}"
                    f"?driver=ODBC+Driver+17+for+SQL+Server"
                )
                
            # Test connection
            try:
                conn = pyodbc.connect(
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"UID={enterprise.sqlserver_admin_user if is_shared else session['username']};"
                    f"PWD={enterprise.sqlserver_admin_password if is_shared else user.sqlserver_password};"
                    f"DATABASE={selected_db}"
                )
                conn.close()
            except Exception as e:
                logger.error(f"SQL Server connection test failed: {str(e)}")
                return jsonify({
                    'error': f'SQL Server connection failed: {str(e)}. '
                            'Please verify: 1) SQL Server allows remote connections, '
                            '2) SQL Server Browser service is running, '
                            '3) Firewall allows port 1433 (or your custom port)'
                }), 400
        elif db_type == 'postgres':
            if not enterprise.postgres_host or not enterprise.postgres_admin_user or not enterprise.postgres_admin_password:
                return jsonify({'error': 'PostgreSQL configuration not found'}), 400
            
            # Verify database exists
            conn_str = f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}" \
                      f"@{enterprise.postgres_host}:{enterprise.postgres_port}/postgres"
            engine = create_engine(conn_str)
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :db_name"), {"db_name": selected_db})
                if not result.fetchone():
                    return jsonify({'error': f"Database '{selected_db}' does not exist in PostgreSQL"}), 404

            if is_shared:
                db_uri = f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}" \
                        f"@{enterprise.postgres_host}:{enterprise.postgres_port}/{selected_db}"
            else:
                if not user.postgres_password:
                    return jsonify({'error': 'PostgreSQL credentials not found for user'}), 401
                db_uri = f"postgresql://{session['username']}:{user.postgres_password}" \
                        f"@{enterprise.postgres_host}:{enterprise.postgres_port}/{selected_db}"
        elif db_type == 'mysql':
            if not enterprise.mysql_host or not enterprise.mysql_admin_user or not enterprise.mysql_admin_password:
                return jsonify({'error': 'MySQL configuration not found'}), 400
            
            if is_shared:
                db_uri = f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}" \
                        f"@{enterprise.mysql_host}:{enterprise.mysql_port}/{selected_db}"
            else:
                if not user.mysql_password:
                    return jsonify({'error': 'MySQL credentials not found for user'}), 401
                db_uri = f"mysql+pymysql://{session['username']}:{user.mysql_password}" \
                        f"@{enterprise.mysql_host}:{enterprise.mysql_port}/{selected_db}"
        else:
            return jsonify({'error': 'Invalid database type specified'}), 400

        # Test the connection
        try:
            engine = create_engine(db_uri)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"Successfully connected to {selected_db} ({db_type})")
        except Exception as e:
            logger.error(f"Connection test failed for {selected_db} ({db_type}): {str(e)}")
            return jsonify({'error': f'Connection test failed: {str(e)}'}), 500

        session['current_database'] = selected_db
        session['connection_type'] = db_type
        return jsonify({'success': True, 'message': f'Connected to {selected_db} ({db_type})'})
    except Exception as e:
        logger.error(f"Error connecting to database {selected_db} ({db_type}): {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    if 'current_database' not in session or 'connection_type' not in session:
        return jsonify({'error': 'No database connected'}), 400

    data = request.get_json()
    user_query = data.get('query')
    thread_id = data.get('thread_id')

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'error': 'Enterprise not found'}), 404

        user = User.query.filter_by(username=session['username']).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        shared_databases = [
            {'name': db.database_name, 'type': db.database_type}
            for db in SharedDatabase.query.filter_by(enterprise_id=enterprise.id).all()
        ]
        is_shared = any(db['name'] == session['current_database'] and db['type'] == session['connection_type'] for db in shared_databases)

        # Initialize sql_db based on database type
        db_uri = None
        if session['connection_type'] == 'sqlserver':
            if not enterprise.sqlserver_host or not enterprise.sqlserver_admin_user or not enterprise.sqlserver_admin_password:
                return jsonify({'error': 'SQL Server configuration not found'}), 400
            
            # Handle both instance and port-based connections
            if enterprise.sqlserver_instance:
                server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}"
                port_part = ""
            else:
                server = enterprise.sqlserver_host
                port_part = f",{enterprise.sqlserver_port}" if enterprise.sqlserver_port else ""
            
            if is_shared:
                db_uri = (
                    f"mssql+pyodbc://{enterprise.sqlserver_admin_user}:{enterprise.sqlserver_admin_password}"
                    f"@{server}{port_part}/{session['current_database']}"
                    f"?driver=ODBC+Driver+17+for+SQL+Server"
                )
            else:
                if not user.sqlserver_password:
                    return jsonify({'error': 'SQL Server credentials not found for user'}), 401
                db_uri = (
                    f"mssql+pyodbc://{session['username']}:{user.sqlserver_password}"
                    f"@{server}{port_part}/{session['current_database']}"
                    f"?driver=ODBC+Driver+17+for+SQL+Server"
                )
        elif session['connection_type'] == 'postgres':
            if not enterprise.postgres_host or not enterprise.postgres_admin_user or not enterprise.postgres_admin_password:
                return jsonify({'error': 'PostgreSQL configuration not found'}), 400
            if is_shared:
                db_uri = f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}" \
                        f"@{enterprise.postgres_host}:{enterprise.postgres_port}/{session['current_database']}"
            else:
                if not user.postgres_password:
                    return jsonify({'error': 'PostgreSQL credentials not found for user'}), 401
                db_uri = f"postgresql://{session['username']}:{user.postgres_password}" \
                        f"@{enterprise.postgres_host}:{enterprise.postgres_port}/{session['current_database']}"
        elif session['connection_type'] == 'mysql':
            if not enterprise.mysql_host or not enterprise.mysql_admin_user or not enterprise.mysql_admin_password:
                return jsonify({'error': 'MySQL configuration not found'}), 400
            if is_shared:
                db_uri = f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}" \
                        f"@{enterprise.mysql_host}:{enterprise.mysql_port}/{session['current_database']}"
            else:
                if not user.mysql_password:
                    return jsonify({'error': 'MySQL credentials not found for user'}), 401
                db_uri = f"mysql+pymysql://{session['username']}:{user.mysql_password}" \
                        f"@{enterprise.mysql_host}:{enterprise.mysql_port}/{session['current_database']}"
        else:
            return jsonify({'error': 'Invalid database type in session'}), 400

        # Create SQLDatabase instance
        try:
            sql_db = SQLDatabase.from_uri(db_uri)
            logger.info(f"Connected to {session['current_database']} ({session['connection_type']})")
        except Exception as e:
            logger.error(f"Failed to connect to database {session['current_database']} ({session['connection_type']}): {str(e)}")
            return jsonify({'error': f'Failed to connect to database: {str(e)}'}), 500

        chat_history = []
        if thread_id:
            try:
                with app.test_client() as client:
                    with client.session_transaction() as sess:
                        sess['username'] = session['username']
                    response = client.get(
                        f"/api/threads/{thread_id}/context",
                        query_string={'database': session['current_database']}
                    )
                    if response.status_code == 200:
                        chat_history = response.json.get('messages', [])
                        logger.info(f"Retrieved {len(chat_history)} messages from chat history")
                    else:
                        logger.warning(f"Failed to get chat history: {response.status_code}")
            except Exception as e:
                logger.warning(f"Error fetching chat history: {str(e)}")
                chat_history = []  # Proceed with empty history if fetch fails
        else:
            logger.info("No thread_id provided; using empty chat history")

        if llm_service is None:
            return jsonify({'error': 'LLM service not initialized'}), 500

        # Check if this is a forecasting request
        is_forecasting_request = any(word in user_query.lower() for word in [
            'forecast', 'predict', 'future', 'next month', 'next year', 'projection',
            'what will be', 'upcoming', 'estimate', 'outlook'
        ])

        if is_forecasting_request:
            try:
                # Get historical data for forecasting
                historical_query = f"Get historical data needed to {user_query}"
                sql_chain = llm_service.get_sql_chain(sql_db, enterprise.id, session['current_database'])
                sql_query = sql_chain.invoke({
                    "question": historical_query,
                    "chat_history": "\n".join([
                        f"{'User' if msg['sender'] == 'user' else 'Assistant'}: {msg['content']}"
                        for msg in chat_history
                    ])
                })
                
                logger.info(f"Forecasting SQL Query: {sql_query}")
                sql_response = sql_db.run(sql_query)
                logger.info(f"Forecasting SQL Response: {sql_response[:200]}...")

                # Parse the response into a DataFrame
                import pandas as pd
                from io import StringIO
                import json
                import ast

                try:
                    data = json.loads(sql_response)
                    df = pd.DataFrame(data)
                except json.JSONDecodeError:
                    try:
                        data = ast.literal_eval(sql_response)
                        df = pd.DataFrame(data)
                    except:
                        df = pd.read_csv(StringIO(sql_response))
                
                logger.info(f"Parsed DataFrame with columns: {df.columns.tolist()}")

                # Detect date and value columns
                date_col = None
                value_col = None
                
                for col in df.columns:
                    col_lower = col.lower()
                    if not date_col and any(word in col_lower for word in ['date', 'time', 'day', 'month', 'year', 'timestamp']):
                        date_col = col
                    elif not value_col and any(word in col_lower for word in ['value', 'amount', 'price', 'sales', 'quantity', 'count', 'total']):
                        value_col = col
                
                if not date_col or not value_col:
                    raise ValueError("Could not automatically detect date and value columns")
                
                logger.info(f"Detected date column: {date_col}, value column: {value_col}")

                # Determine forecast periods from query
                periods = 5
                import re
                period_match = re.search(r'next (\d+) (day|week|month|year)', user_query.lower())
                if period_match:
                    periods = int(period_match.group(1))
                    if period_match.group(2) == 'week':
                        periods *= 7
                    elif period_match.group(2) == 'month':
                        periods *= 30
                    elif period_match.group(2) == 'year':
                        periods *= 365
                
                # Perform forecasting
                forecast_result = llm_service.forecasting_service.forecast(
                    df.to_dict('records'),
                    date_col=date_col,
                    value_col=value_col,
                    periods=periods,
                    model='auto'
                )
                
                # Format the forecast response
                forecast_text = llm_service.format_forecast_results(forecast_result, user_query)
                chart_data = llm_service.create_forecast_visualization(df, forecast_result, date_col, value_col)
                
                # Store the message in the database with chart data
                with app.test_client() as client:
                    with client.session_transaction() as sess:
                        sess['username'] = session['username']
                    response = client.post(
                        f"/api/threads/{thread_id}/messages",
                        json={
                            'sender': 'bot',
                            'content': forecast_text,
                            'database': session['current_database'],
                            'chart_data': chart_data
                        }
                    )
                    if response.status_code != 200:
                        logger.warning(f"Failed to save bot message: {response.status_code}")

                return jsonify({
                    "text_response": forecast_text,
                    "chart_data": chart_data
                })

            except Exception as e:
                logger.error(f"Forecasting failed: {str(e)}")
                # Fall through to regular processing if forecasting fails

        # Regular non-forecasting processing
        final_nl_response, visualization = llm_service.get_response(
            user_query=user_query,
            db=sql_db,
            chat_history=chat_history,
            enterprise_id=enterprise.id,
            database_name=session['current_database']
        )

        # Store the message in the database with chart data
        with app.test_client() as client:
            with client.session_transaction() as sess:
                sess['username'] = session['username']
            response = client.post(
                f"/api/threads/{thread_id}/messages",
                json={
                    'sender': 'bot',
                    'content': final_nl_response,
                    'database': session['current_database'],
                    'chart_data': visualization
                }
            )
            if response.status_code != 200:
                logger.warning(f"Failed to save bot message: {response.status_code}")

        return jsonify({
            "text_response": final_nl_response,
            "chart_data": visualization
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    

@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        if llm_service is None:
            return jsonify({'error': 'LLM service not initialized'})

        vectorstore = llm_service.process_pdf(file)
        session['current_vectorstore'] = vectorstore

        return jsonify({'success': True, 'message': 'PDF processed successfully'})

    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat_with_pdf', methods=['POST'])
def chat_with_pdf():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    if 'current_vectorstore' not in session:
        return jsonify({'error': 'No PDF has been processed'}),  # 400

    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        return jsonify({'error': 'Missing query'}), 400

    try:
        if llm_service is None:
            return jsonify({'error': 'LLM service not initialized'}), 500

        response, _ = llm_service.get_response(
            user_query,
            vectorstore=session['current_vectorstore']
        )

        return jsonify({'response': response})

    except Exception as e:
        logger.error(f"Error in chat_with_pdf endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/create-database', methods=['POST'])
@require_enterprise_auth
def api_create_database():
    try:
        data = request.get_json()
        if not data or 'database_name' not in data or 'type' not in data:
            return jsonify({'success': False, 'error': 'Missing database name or type'}), 400

        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'success': False, 'error': 'Enterprise not found'}), 404

        db_name = data['database_name']
        db_type = data['type']

        # Validate database name
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]{2,63}$', db_name):
            return jsonify({'success': False, 'error': 'Invalid database name format'}), 400

        user = User.query.filter_by(username=session['username']).first()
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        if db_type == 'sqlserver':
            if not enterprise.sqlserver_host or not enterprise.sqlserver_admin_user or not enterprise.sqlserver_admin_password:
                return jsonify({'success': False, 'error': 'SQL Server configuration not found'}), 400
            
            # Create server string based on whether instance name is provided
            server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
            
            # Create connection string for master database
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"UID={enterprise.sqlserver_admin_user};"
                f"PWD={enterprise.sqlserver_admin_password};"
                f"DATABASE=master;"
                f"Connection Timeout=30;"
            )
            
            conn = pyodbc.connect(conn_str)
            conn.autocommit = True
            cursor = conn.cursor()
            
            try:
                # Create database
                cursor.execute(f"CREATE DATABASE [{db_name}]")
                
                # Check if login exists
                cursor.execute("SELECT 1 FROM sys.server_principals WHERE name = ?", user.username)
                if not cursor.fetchone():
                    # Create login if it doesn't exist
                    cursor.execute(
                        "CREATE LOGIN [{}] WITH PASSWORD = ?, CHECK_EXPIRATION=OFF, CHECK_POLICY=OFF".format(user.username),
                        user.sqlserver_password
                    )
                
                # Grant privileges
                cursor.execute(f"USE [{db_name}]; CREATE USER [{user.username}] FOR LOGIN [{user.username}]")
                cursor.execute(f"USE [{db_name}]; ALTER ROLE db_owner ADD MEMBER [{user.username}]")
                
            finally:
                cursor.close()
                conn.close()
        elif db_type == 'postgres':
            if not enterprise.postgres_host or not enterprise.postgres_admin_user or not enterprise.postgres_admin_password:
                return jsonify({'success': False, 'error': 'PostgreSQL configuration not found'}), 400
            
            # Create PostgreSQL connection
            admin_conn = psycopg2.connect(
                host=enterprise.postgres_host,
                port=enterprise.postgres_port,
                user=enterprise.postgres_admin_user,
                password=enterprise.postgres_admin_password,
                dbname="postgres"
            )
            admin_conn.autocommit = True
            admin_cursor = admin_conn.cursor()
            
            try:
                # Create database
                admin_cursor.execute(f"CREATE DATABASE {db_name}")
                
                # Create user if not exists and set password
                admin_cursor.execute(f"SELECT 1 FROM pg_roles WHERE rolname = '{user.username}'")
                if not admin_cursor.fetchone():
                    admin_cursor.execute(f"CREATE USER {user.username} WITH PASSWORD '{user.postgres_password}'")
                
                # Grant privileges on database
                admin_cursor.execute(f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {user.username}")
                
                # Connect to the newly created database to grant schema privileges
                db_conn = psycopg2.connect(
                    host=enterprise.postgres_host,
                    port=enterprise.postgres_port,
                    user=enterprise.postgres_admin_user,
                    password=enterprise.postgres_admin_password,
                    dbname=db_name
                )
                db_conn.autocommit = True
                db_cursor = db_conn.cursor()
                
                try:
                    # Grant usage and create privileges on public schema
                    db_cursor.execute(f"GRANT USAGE, CREATE ON SCHEMA public TO {user.username}")
                    # Grant all privileges on all tables in public schema (current and future)
                    db_cursor.execute(f"GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO {user.username}")
                    db_cursor.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {user.username}")
                    # Grant all privileges on all sequences in public schema (current and future)
                    db_cursor.execute(f"GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO {user.username}")
                    db_cursor.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO {user.username}")
                finally:
                    db_cursor.close()
                    db_conn.close()
                
            finally:
                admin_cursor.close()
                admin_conn.close()
                
        elif db_type == 'mysql':
            if not enterprise.mysql_host or not enterprise.mysql_admin_user or not enterprise.mysql_admin_password:
                return jsonify({'success': False, 'error': 'MySQL configuration not found'}), 400
            
            # Create MySQL connection using admin credentials
            engine = create_engine(
                f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}"
                f"@{enterprise.mysql_host}:{enterprise.mysql_port}"
            )
            
            with engine.connect() as conn:
                # Create database
                conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))
                
                # Create user if not exists (MySQL syntax)
                conn.execute(text(f"CREATE USER IF NOT EXISTS '{user.username}'@'%' IDENTIFIED BY '{user.mysql_password}'"))
                
                # Grant all privileges on the specific database to the user
                conn.execute(text(f"GRANT ALL PRIVILEGES ON `{db_name}`.* TO '{user.username}'@'%'"))
                
                # Flush privileges to ensure changes take effect
                conn.execute(text("FLUSH PRIVILEGES"))
                
        else:
            return jsonify({'success': False, 'error': 'Invalid database type'}), 400

        # Update user's access list
        access_list = json.loads(user.access) if user.access else []
        if not any(acc['name'] == db_name and acc['type'] == db_type for acc in access_list):
            access_list.append({'name': db_name, 'type': db_type})
            user.access = json.dumps(access_list)
            db.session.commit()

        return jsonify({
            'success': True,
            'message': f'Database {db_name} ({db_type}) created successfully with full permissions',
            'database': db_name
        })

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating database: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error creating database: {str(e)}'
        }), 500



@app.route('/api/database-tables', methods=['POST'])
def get_database_tables():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.get_json()
    selected_db = data.get('database')
    db_type = data.get('type')  # Expect 'mysql', 'postgres', or 'sqlserver'

    if not selected_db or not db_type:
        return jsonify({'error': 'Database name and type are required'}), 400

    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'error': 'Enterprise not found'}), 404

        # Check if database is shared
        is_shared = SharedDatabase.query.filter_by(
            database_name=selected_db,
            database_type=db_type,
            enterprise_id=enterprise.id
        ).first() is not None

        tables = []
        
        if db_type == 'postgres':
            if not all([enterprise.postgres_host, enterprise.postgres_admin_user, enterprise.postgres_admin_password]):
                return jsonify({'error': 'PostgreSQL not configured'}), 400

            if is_shared:
                # Use admin credentials for shared databases
                db_uri = f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}@{enterprise.postgres_host}:{enterprise.postgres_port}/{selected_db}"
            else:
                # Use user credentials for personal databases
                user = User.query.filter_by(username=session['username']).first()
                if not user or not user.postgres_password:
                    return jsonify({'error': 'Missing PostgreSQL credentials'}), 401
                db_uri = f"postgresql://{session['username']}:{user.postgres_password}@{enterprise.postgres_host}:{enterprise.postgres_port}/{selected_db}"

            engine = create_engine(db_uri)
            with engine.connect() as conn:
                # Get tables from public schema
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_type = 'BASE TABLE'
                """))
                tables = [row[0] for row in result]
        
        elif db_type == 'mysql':
            if not enterprise.mysql_host or not enterprise.mysql_admin_user or not enterprise.mysql_admin_password:
                return jsonify({'error': 'MySQL configuration not found'}), 400
            
            # Use admin credentials for shared databases, user credentials for personal
            if is_shared:
                db_uri = f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}" \
                        f"@{enterprise.mysql_host}:{enterprise.mysql_port}/{selected_db}"
            else:
                user = User.query.filter_by(username=session['username']).first()
                if not user or not user.mysql_password:
                    return jsonify({'error': 'MySQL credentials not found for user'}), 401
                db_uri = f"mysql+pymysql://{session['username']}:{user.mysql_password}" \
                        f"@{enterprise.mysql_host}:{enterprise.mysql_port}/{selected_db}"
            
            # Try connection with retries
            engine = create_engine(db_uri)
            try:
                with engine.connect() as conn:
                    result = conn.execute(text("SHOW TABLES"))
                    tables = [row[0] for row in result]
            except Exception as e:
                logger.error(f"MySQL connection error: {str(e)}")
                return jsonify({'error': f'MySQL connection failed: {str(e)}'}), 500

        elif db_type == 'sqlserver':
            if not all([enterprise.sqlserver_host, enterprise.sqlserver_admin_user, enterprise.sqlserver_admin_password]):
                return jsonify({'error': 'SQL Server not configured'}), 400

            # Handle both instance and port-based connections
            if enterprise.sqlserver_instance:
                server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}"
            else:
                server = f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}" if enterprise.sqlserver_port else enterprise.sqlserver_host

            if is_shared:
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"UID={enterprise.sqlserver_admin_user};"
                    f"PWD={enterprise.sqlserver_admin_password};"
                    f"DATABASE={selected_db}"
                )
            else:
                user = User.query.filter_by(username=session['username']).first()
                if not user or not user.sqlserver_password:
                    return jsonify({'error': 'SQL Server credentials not found'}), 401
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"UID={session['username']};"
                    f"PWD={user.sqlserver_password};"
                    f"DATABASE={selected_db}"
                )

            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
            tables = [row.TABLE_NAME for row in cursor]
            cursor.close()
            conn.close()

        return jsonify({'tables': tables})

    except Exception as e:
        logger.error(f"Error fetching tables for database {selected_db} ({db_type}): {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/threads/<int:thread_id>/export', methods=['GET'])
def export_thread(thread_id):
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        thread = ChatThread.query.filter_by(id=thread_id, user_id=user.id).first()
        if not thread:
            return jsonify({'error': 'Thread not found or access denied'}), 404

        messages = ChatMessage.query.filter_by(thread_id=thread_id).order_by(ChatMessage.timestamp.asc()).all()
        
        # Format messages into a string
        content = f"Chat Thread: {thread.title}\nCreated: {thread.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        for msg in messages:
            sender = 'User' if msg.sender == 'user' else 'Bot'
            timestamp = msg.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            content += f"[{timestamp}] {sender}: {msg.content}\n\n"

        # Create a file-like object
        file_stream = io.StringIO(content)
        file_stream.seek(0)

        # Generate a safe filename
        safe_title = secure_filename(thread.title.replace(' ', '_'))
        filename = f"chat_{safe_title}_{thread_id}.txt"

        # Log the export action
        log_admin_action(session['username'], 'export_chat', f'Exported chat thread {thread_id}: {thread.title}')

        return send_file(
            io.BytesIO(content.encode('utf-8')),
            mimetype='text/plain',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"Error exporting thread {thread_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


from flask import Blueprint, request, jsonify, session, flash, redirect, url_for, render_template
from flask_login import login_user, logout_user, login_required, current_user


main = Blueprint('main', __name__)

@main.route('/api/threads/<int:thread_id>/message-pair', methods=['GET'])
@login_required
def get_message_pair(thread_id):
    bot_timestamp = request.args.get('bot_timestamp')
    if not bot_timestamp:
        return jsonify({'error': 'Bot timestamp is required'}), 400

    try:
        # Fetch the bot message
        bot_message = ChatMessage.query.filter_by(
            thread_id=thread_id,
            sender='bot',
            timestamp=bot_timestamp
        ).first()

        if not bot_message:
            return jsonify({'error': 'Bot message not found'}), 404

        # Fetch the preceding user message (if any)
        user_message = ChatMessage.query.filter_by(
            thread_id=thread_id,
            sender='user'
        ).filter(ChatMessage.timestamp < bot_timestamp).order_by(ChatMessage.timestamp.desc()).first()

        messages = []
        if user_message:
            messages.append({
                'sender': user_message.sender,
                'content': user_message.content,
                'timestamp': user_message.timestamp.isoformat(),
                'database': user_message.database
            })
        messages.append({
            'sender': bot_message.sender,
            'content': bot_message.content,
            'timestamp': bot_message.timestamp.isoformat(),
            'database': bot_message.database
        })

        return jsonify({'messages': messages}), 200
    except Exception as e:
        print(f"Error fetching message pair: {e}")
        return jsonify({'error': 'Failed to fetch message pair'}), 500


@app.route('/api/delete-database', methods=['POST'])
def delete_database():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.get_json()
    selected_db = data.get('database')
    db_type = data.get('type')  # Expect 'mysql', 'postgres', or 'sqlserver'

    if not selected_db or not db_type:
        return jsonify({'error': 'Database name and type are required'}), 400

    try:
        # Verify user and access
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Parse access field as JSON
        try:
            access_list = json.loads(user.access) if user.access else []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in user.access for user {session['username']}: {user.access}")
            return jsonify({'error': 'Invalid access configuration'}), 500

        # Check if user has access to the database with the specified type
        has_access = any(db['name'] == selected_db and db['type'] == db_type for db in access_list)
        logger.info(f"User: {session['username']}, Access list: {access_list}, Selected DB: {selected_db}, DB Type: {db_type}, Has Access: {has_access}")
        if not has_access:
            return jsonify({'error': 'No access to selected database'}), 403

        # Check if database is shared
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'error': 'Enterprise not found'}), 404

        shared_databases = [
            {'name': db.database_name, 'type': db.database_type}
            for db in SharedDatabase.query.filter_by(enterprise_id=enterprise.id).all()
        ]
        if any(db['name'] == selected_db and db['type'] == db_type for db in shared_databases):
            return jsonify({'error': 'Cannot delete shared database'}), 403

        # Drop the database based on db_type
        if db_type == 'mysql':
            if not enterprise.mysql_host or not enterprise.mysql_admin_user or not enterprise.mysql_admin_password:
                return jsonify({'error': 'MySQL configuration not found'}), 400
            admin_engine = create_engine(
                f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}"
                f"@{enterprise.mysql_host}:{enterprise.mysql_port}"
            )
            with admin_engine.connect() as conn:
                conn.execute(text(f"DROP DATABASE IF EXISTS `{selected_db}`"))
                logger.info(f"Dropped MySQL database {selected_db} for user {session['username']}")
        elif db_type == 'postgres':
            if not enterprise.postgres_host or not enterprise.postgres_admin_user or not enterprise.postgres_admin_password:
                return jsonify({'error': 'PostgreSQL configuration not found'}), 400
            admin_engine = create_engine(
                f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}"
                f"@{enterprise.postgres_host}:{enterprise.postgres_port}/postgres",
                isolation_level='AUTOCOMMIT'  # Set autocommit to avoid transaction block
            )
            with admin_engine.connect() as conn:
                # Terminate active connections
                conn.execute(
                    text(
                        f"SELECT pg_terminate_backend(pg_stat_activity.pid) "
                        f"FROM pg_stat_activity "
                        f"WHERE pg_stat_activity.datname = :db_name AND pid <> pg_backend_pid()"
                    ),
                    {"db_name": selected_db}
                )
                conn.execute(text(f"DROP DATABASE IF EXISTS \"{selected_db}\""))
                logger.info(f"Dropped PostgreSQL database {selected_db} for user {session['username']}")
        elif db_type == 'sqlserver':
            if not enterprise.sqlserver_host or not enterprise.sqlserver_admin_user or not enterprise.sqlserver_admin_password:
                return jsonify({'error': 'SQL Server configuration not found'}), 400
            server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"UID={enterprise.sqlserver_admin_user};"
                f"PWD={enterprise.sqlserver_admin_password};"
            )
            conn = pyodbc.connect(conn_str)
            conn.autocommit = True
            cursor = conn.cursor()
            # Set database to single-user mode to terminate connections
            cursor.execute(f"ALTER DATABASE [{selected_db}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE")
            cursor.execute(f"DROP DATABASE IF EXISTS [{selected_db}]")
            cursor.close()
            conn.close()
            logger.info(f"Dropped SQL Server database {selected_db} for user {session['username']}")
        else:
            return jsonify({'error': 'Invalid database type specified'}), 400

        # Update user's access list
        access_list = [db for db in access_list if not (db['name'] == selected_db and db['type'] == db_type)]
        user.access = json.dumps(access_list) if access_list else ''
        db.session.commit()

        # Log the action
        log_admin_action(session['username'], 'delete_database', f'Deleted database {selected_db} ({db_type})')

        return jsonify({'success': True, 'message': f'Database {selected_db} ({db_type}) deleted successfully'})

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting database {selected_db} ({db_type}): {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-table', methods=['POST'])
def delete_table():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.get_json()
    selected_db = data.get('database')
    table_name = data.get('table')
    db_type = data.get('type')  # Expect 'mysql', 'postgres', or 'sqlserver'

    if not selected_db or not table_name or not db_type:
        return jsonify({'error': 'Database, table name, and type are required'}), 400

    try:
        # Verify user and access
        user = User.query.filter_by(username=session['username']).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        # Parse access field as JSON
        try:
            access_list = json.loads(user.access) if user.access else []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in user.access for user {session['username']}: {user.access}")
            return jsonify({'error': 'Invalid access configuration'}), 500

        # Check if user has access to the database with the specified type
        has_access = any(db['name'] == selected_db and db['type'] == db_type for db in access_list)
        logger.info(f"User: {session['username']}, Access list: {access_list}, Selected DB: {selected_db}, DB Type: {db_type}, Has Access: {has_access}")
        if not has_access:
            return jsonify({'error': 'No access to selected database'}), 403

        # Check if database is shared
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'error': 'Enterprise not found'}), 404

        shared_databases = [
            {'name': db.database_name, 'type': db.database_type}
            for db in SharedDatabase.query.filter_by(enterprise_id=enterprise.id).all()
        ]
        if any(db['name'] == selected_db and db['type'] == db_type for db in shared_databases):
            return jsonify({'error': 'Cannot delete tables in shared database'}), 403

        # Drop the table using appropriate credentials
        if db_type == 'mysql':
            if not enterprise.mysql_host or not enterprise.mysql_admin_user or not enterprise.mysql_admin_password:
                return jsonify({'error': 'MySQL configuration not found'}), 400
            if not user.mysql_password:
                return jsonify({'error': 'MySQL credentials not found for user'}), 401
            db_uri = f"mysql+pymysql://{session['username']}:{user.mysql_password}" \
                    f"@{enterprise.mysql_host}:{enterprise.mysql_port}/{selected_db}"
            engine = create_engine(db_uri)
            with engine.connect() as conn:
                escaped_table = table_name.replace("'", "''")
                conn.execute(text(f"DROP TABLE IF EXISTS `{escaped_table}`"))
                logger.info(f"Dropped MySQL table {table_name} in database {selected_db} for user {session['username']}")
        elif db_type == 'postgres':
            if not enterprise.postgres_host or not enterprise.postgres_admin_user or not enterprise.postgres_admin_password:
                return jsonify({'error': 'PostgreSQL configuration not found'}), 400
            if not user.postgres_password:
                return jsonify({'error': 'PostgreSQL credentials not found for user'}), 401
            db_uri = f"postgresql://{session['username']}:{user.postgres_password}" \
                    f"@{enterprise.postgres_host}:{enterprise.postgres_port}/{selected_db}"
            engine = create_engine(db_uri, isolation_level='AUTOCOMMIT')
            try:
                with engine.connect() as conn:
                    escaped_table = table_name.replace('"', '""')
                    sql_command = f'DROP TABLE IF EXISTS "{escaped_table}"'
                    logger.debug(f"Executing PostgreSQL command: {sql_command}")
                    conn.execute(text(sql_command))
                    conn.execute(text("COMMIT"))  # Explicit commit to ensure changes are applied
                    logger.info(f"Dropped PostgreSQL table {table_name} in database {selected_db} for user {session['username']}")
            except Exception as e:
                logger.error(f"Failed to drop PostgreSQL table {table_name} in database {selected_db}: {str(e)}")
                raise
        elif db_type == 'sqlserver':
            if not enterprise.sqlserver_host or not enterprise.sqlserver_admin_user or not enterprise.sqlserver_admin_password:
                return jsonify({'error': 'SQL Server configuration not found'}), 400
            if not user.sqlserver_password:
                return jsonify({'error': 'SQL Server credentials not found for user'}), 401
            server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"UID={session['username']};"
                f"PWD={user.sqlserver_password};"
                f"DATABASE={selected_db}"
            )
            conn = pyodbc.connect(conn_str)
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS [{table_name}]")
            cursor.close()
            conn.close()
            logger.info(f"Dropped SQL Server table {table_name} in database {selected_db} for user {session['username']}")
        else:
            return jsonify({'error': 'Invalid database type specified'}), 400

        # Log the action
        log_admin_action(session['username'], 'delete_table', f'Deleted table {table_name} in database {selected_db} ({db_type})')

        return jsonify({'success': True, 'message': f'Table {table_name} deleted successfully'})

    except Exception as e:
        logger.error(f"Error deleting table {table_name} in database {selected_db} ({db_type}): {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/current-enterprise')
def api_current_enterprise():
    if 'username' not in session:  # Use session check instead of login_required
        return jsonify({'error': 'Not logged in'}), 401

    user = User.query.filter_by(username=session['username']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({
        'enterprise_id': user.enterprise_id,
        'enterprise_name': Enterprise.query.get(user.enterprise_id).name if user.enterprise_id else None
    })
    
@app.route('/forecast')
def forecast_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('forecast.html')

@app.route('/api/forecast', methods=['POST'])
def api_forecast():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Get the required fields with validation
        required_fields = ['data', 'date_col', 'value_col']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Get forecasting parameters with defaults
        periods = int(data.get('periods', 5))
        model = data.get('model', 'auto')
        row_limit = data.get('row_limit', len(data['data']))  # Use all rows by default
        feature_cols = data.get('feature_cols', [])

        # Use all provided data (already limited by the table-preview endpoint)
        limited_data = data['data']

        # Initialize forecasting service
        forecasting_service = ForecastingService()

        # Run forecast with feature columns
        result = forecasting_service.forecast(
            limited_data,
            data['date_col'],
            data['value_col'],
            periods,
            model,
            feature_cols
        )

        # Format the results for the frontend
        response = {
            'success': True,
            'model': result['model'],
            'forecast': result['forecast'],
            'last_date': str(result['last_date']),
            'metrics': result.get('metrics', {}),
            'row_limit': row_limit
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Forecasting error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500


from flask import make_response
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io
import base64
import re

# Add this new route
@app.route('/api/export-forecast', methods=['POST'])
def export_forecast():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Add title
        title = Paragraph("Forecast Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))

        # Add generation date
        date_text = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                            styles['Normal'])
        story.append(date_text)
        story.append(Spacer(1, 24))

        # Add input parameters section
        inputs_title = Paragraph("Input Parameters", styles['Heading2'])
        story.append(inputs_title)
        
        inputs = [
            f"<b>Database:</b> {data.get('database', 'N/A')}",
            f"<b>Table:</b> {data.get('table', 'N/A')}",
            f"<b>Date Column:</b> {data.get('date_col', 'N/A')}",
            f"<b>Value Column:</b> {data.get('value_col', 'N/A')}",
            f"<b>Forecast Periods:</b> {data.get('periods', 'N/A')}",
            f"<b>Model:</b> {data.get('model', 'N/A')}"
        ]
        
        for item in inputs:
            story.append(Paragraph(item, styles['Normal']))
            story.append(Spacer(1, 6))
        
        story.append(Spacer(1, 12))

        # Add forecast results if available
        if data.get('results'):
            results_title = Paragraph("Forecast Results", styles['Heading2'])
            story.append(results_title)
            
            model_info = Paragraph(f"<b>Model Used:</b> {data.get('model_info', 'N/A')}", 
                                 styles['Normal'])
            story.append(model_info)
            story.append(Spacer(1, 6))
            
            # Add forecast values
            results = data['results'].split('\n')
            for result in results:
                if result.strip():
                    story.append(Paragraph(result.strip(), styles['Normal']))
                    story.append(Spacer(1, 6))
            
            story.append(Spacer(1, 12))

            # Add chart if available
            if data.get('chart_image'):
                try:
                    # Decode base64 image
                    chart_data = re.sub('^data:image/.+;base64,', '', data['chart_image'])
                    chart_bytes = base64.b64decode(chart_data)
                    chart_img = Image(io.BytesIO(chart_bytes), width=6*inch, height=4*inch)
                    story.append(chart_img)
                except Exception as e:
                    error_msg = Paragraph("<b>Note:</b> Could not include chart in export", 
                                        styles['Italic'])
                    story.append(error_msg)

        # Build PDF
        doc.build(story)
        buffer.seek(0)

        # Create response
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = \
            f'attachment; filename=forecast_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/test-mysql-connection', methods=['POST'])
def test_mysql_connection():
    try:
        data = request.get_json()
        connection_string = f"mysql+pymysql://{data['user']}:{data['password']}@{data['host']}:{data['port']}"
        
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save-mysql-config', methods=['POST'])
def save_mysql_config():
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['host', 'port', 'user', 'password']
        if not all(field in data for field in required_fields):
            return jsonify({'success': False, 'error': 'Missing required fields'})
        
        # Test the connection first
        connection_string = f"mysql+pymysql://{data['user']}:{data['password']}@{data['host']}:{data['port']}"
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            return jsonify({'success': False, 'error': f'Connection test failed: {str(e)}'})
        
        # Save the config
        with open(MYSQL_CONFIG_FILE, 'w') as f:
            json.dump(data, f)
        
        # Update app config
        app.config['MYSQL_HOST'] = data['host']
        app.config['MYSQL_PORT'] = data['port']
        app.config['MYSQL_ADMIN_USER'] = data['user']
        app.config['MYSQL_ADMIN_PASSWORD'] = data['password']
        
        # Get all non-system databases
        with engine.connect() as conn:
            result = conn.execute(text("SHOW DATABASES"))
            system_dbs = {
                'information_schema',
                'mysql',
                'performance_schema',
                'sys',
                'admin'
            }
            all_databases = [row[0] for row in result if row[0] not in system_dbs]
            
            # Get current shared databases
            shared_dbs = [db.database_name for db in SharedDatabase.query.all()]
            
            return jsonify({
                'success': True,
                'databases': all_databases,
                'shared_databases': shared_dbs,
                'message': 'Configuration saved successfully'
            })
            
    except Exception as e:
        logger.error(f"Error saving MySQL config: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error saving configuration: {str(e)}'
        })
        
@app.route('/api/get-mysql-config', methods=['GET'])
def get_mysql_config():
    try:
        config = load_db_configs()
        if config:
            return jsonify({
                'success': True,
                'config': config
            })
        return jsonify({
            'success': False,
            'message': 'No MySQL configuration found'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
        
def load_db_configs():
    try:
        configs = {}
        if os.path.exists(MYSQL_CONFIG_FILE):
            with open(MYSQL_CONFIG_FILE, 'r') as f:
                configs['mysql'] = json.load(f)
                app.config['MYSQL_HOST'] = configs['mysql'].get('host')
                app.config['MYSQL_PORT'] = configs['mysql'].get('port', 3306)
                app.config['MYSQL_ADMIN_USER'] = configs['mysql'].get('user')
                app.config['MYSQL_ADMIN_PASSWORD'] = configs['mysql'].get('password')
        if os.path.exists(POSTGRES_CONFIG_FILE):
            with open(POSTGRES_CONFIG_FILE, 'r') as f:
                configs['postgres'] = json.load(f)
                app.config['POSTGRES_HOST'] = configs['postgres'].get('host')
                app.config['POSTGRES_PORT'] = configs['postgres'].get('port', 5432)
                app.config['POSTGRES_ADMIN_USER'] = configs['postgres'].get('user')
                app.config['POSTGRES_ADMIN_PASSWORD'] = configs['postgres'].get('password')
        return configs if configs else None
    except Exception as e:
        logger.error(f"Error loading database configs: {str(e)}")
        return None
    
@app.route('/api/check-mysql-config', methods=['GET'])
def check_mysql_config():
    config = load_db_configs()
    return jsonify({
        'has_config': config is not None,
        'config': {
            'host': config.get('host') if config else '',
            'port': config.get('port') if config else '',
            'user': config.get('user') if config else '',
            # Never return password
        } if config else {}
    })
    
    
@app.route('/api/export-forecast-csv', methods=['POST'])
def export_forecast_csv():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Create CSV content
        csv_content = "Date,Forecast Value\n"
        
        model = data.get('model', 'ARIMA')
        forecast = data.get('forecast', [])
        last_date = data.get('last_date')
        
        if not forecast:
            return jsonify({'error': 'No forecast data to export'}), 400

        try:
            if model == 'ARIMA':
                # Handle last_date which might include time portion
                last_date_str = last_date.split('T')[0] if 'T' in last_date else last_date.split(' ')[0]
                last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
                for i, value in enumerate(forecast, 1):
                    forecast_date = last_date + timedelta(days=i)
                    csv_content += f"{forecast_date.strftime('%Y-%m-%d')},{value}\n"
            elif model == 'Random Forest':
                for item in forecast:
                    date_str = item.get('date', '')
                    # Handle multiple possible date formats
                    if isinstance(date_str, str):
                        try:
                            # Try ISO format first
                            date = datetime.fromisoformat(date_str)
                        except ValueError:
                            try:
                                # Try RFC 1123 format (e.g., "Fri, 10 Apr 2020 22:35:08 GMT")
                                date = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
                            except ValueError:
                                try:
                                    # Try SQL format (e.g., "2020-04-10 22:35:08")
                                    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                                except ValueError:
                                    # Fallback to just date portion
                                    date_part = date_str.split('T')[0] if 'T' in date_str else date_str.split(' ')[0]
                                    date = datetime.strptime(date_part, '%Y-%m-%d')
                    else:
                        date = date_str
                    csv_content += f"{date.strftime('%Y-%m-%d')},{item.get('prediction', '')}\n"
            else:
                return jsonify({'error': f'Unsupported model type: {model}'}), 400

        except Exception as format_error:
            logger.error(f"Error formatting CSV data: {str(format_error)}")
            return jsonify({'error': f'Error formatting data: {str(format_error)}'}), 500

        # Create response
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = \
            f'attachment; filename=forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response

    except Exception as e:
        logger.error(f"Error in CSV export: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/table-preview', methods=['POST'])
def get_table_preview():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.get_json()
    selected_db = data.get('database')
    table_name = data.get('table')
    row_limit = data.get('limit')  # None means no limit
    order_by = data.get('order_by')
    order_direction = data.get('order_direction', 'ASC')
    db_type = data.get('type')  # Expect 'mysql', 'postgres', or 'sqlserver'

    if not selected_db or not table_name or not db_type:
        return jsonify({'error': 'Database, table name, and type are required'}), 400

    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'error': 'Enterprise not found'}), 404

        shared_databases = [
            {'name': db.database_name, 'type': db.database_type}
            for db in SharedDatabase.query.filter_by(enterprise_id=enterprise.id).all()
        ]
        is_shared = any(db['name'] == selected_db and db['type'] == db_type for db in shared_databases)

        # Initialize columns and rows with default empty values
        columns = []
        rows = []
        
        if db_type == 'postgres':
            if not enterprise.postgres_host or not enterprise.postgres_admin_user or not enterprise.postgres_admin_password:
                return jsonify({'error': 'PostgreSQL configuration not found'}), 400
            if is_shared:
                db_uri = f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}" \
                        f"@{enterprise.postgres_host}:{enterprise.postgres_port}/{selected_db}"
            else:
                user = User.query.filter_by(username=session['username']).first()
                if not user or not user.postgres_password:
                    return jsonify({'error': 'PostgreSQL credentials not found for user'}), 401
                db_uri = f"postgresql://{session['username']}:{user.postgres_password}" \
                        f"@{enterprise.postgres_host}:{enterprise.postgres_port}/{selected_db}"
            
            engine = create_engine(db_uri)
            with engine.connect() as conn:
                escaped_table = table_name.replace("'", "''")
                order_clause = f"ORDER BY \"{order_by}\" {order_direction}" if order_by else ""
                limit_clause = f"LIMIT {row_limit}" if row_limit else ""
                query = text(f"SELECT * FROM \"{escaped_table}\" {order_clause} {limit_clause}")
                result = conn.execute(query)
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]

        elif db_type == 'mysql':
            if not enterprise.mysql_host or not enterprise.mysql_admin_user or not enterprise.mysql_admin_password:
                return jsonify({'error': 'MySQL configuration not found'}), 400
            if is_shared:
                db_uri = f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}" \
                        f"@{enterprise.mysql_host}:{enterprise.mysql_port}/{selected_db}"
            else:
                user = User.query.filter_by(username=session['username']).first()
                if not user or not user.mysql_password:
                    return jsonify({'error': 'MySQL credentials not found for user'}), 401
                db_uri = f"mysql+pymysql://{session['username']}:{user.mysql_password}" \
                        f"@{enterprise.mysql_host}:{enterprise.mysql_port}/{selected_db}"
            
            engine = create_engine(db_uri)
            with engine.connect() as conn:
                escaped_table = table_name.replace("'", "''")
                order_clause = f"ORDER BY `{order_by}` {order_direction}" if order_by else ""
                limit_clause = f"LIMIT {row_limit}" if row_limit else ""
                query = text(f"SELECT * FROM `{escaped_table}` {order_clause} {limit_clause}")
                result = conn.execute(query)
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]

        elif db_type == 'sqlserver':
            if not all([enterprise.sqlserver_host, enterprise.sqlserver_admin_user, enterprise.sqlserver_admin_password]):
                return jsonify({'error': 'SQL Server not configured'}), 400

            # Handle both instance and port-based connections
            if enterprise.sqlserver_instance:
                server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}"
            else:
                server = f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}" if enterprise.sqlserver_port else enterprise.sqlserver_host

            if is_shared:
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"UID={enterprise.sqlserver_admin_user};"
                    f"PWD={enterprise.sqlserver_admin_password};"
                    f"DATABASE={selected_db}"
                )
            else:
                user = User.query.filter_by(username=session['username']).first()
                if not user or not user.sqlserver_password:
                    return jsonify({'error': 'SQL Server credentials not found for user'}), 401
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"UID={session['username']};"
                    f"PWD={user.sqlserver_password};"
                    f"DATABASE={selected_db}"
                )
            
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            order_clause = f"ORDER BY [{order_by}] {order_direction}" if order_by else ""
            limit_clause = f"TOP {row_limit}" if row_limit else ""
            query = f"SELECT {limit_clause} * FROM [{table_name}] {order_clause}"
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            conn.close()

        return jsonify({
            'columns': list(columns) if columns else [],
            'rows': rows if rows else []
        })

    except Exception as e:
        logger.error(f"Error fetching preview for table {table_name} in database {selected_db}: {str(e)}")
        return jsonify({
            'error': str(e),
            'columns': [],
            'rows': []
        }), 500
    
@app.route('/api/table-preview10', methods=['POST'])
def get_table_preview10():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.get_json()
    selected_db = data.get('database')
    table_name = data.get('table')
    db_type = data.get('type')  # Expect 'mysql', 'postgres', or 'sqlserver'

    if not selected_db or not table_name or not db_type:
        return jsonify({'error': 'Database, table name, and type are required'}), 400

    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'error': 'Enterprise not found'}), 404

        shared_databases = [
            {'name': db.database_name, 'type': db.database_type}
            for db in SharedDatabase.query.filter_by(enterprise_id=enterprise.id).all()
        ]
        is_shared = any(db['name'] == selected_db and db['type'] == db_type for db in shared_databases)

        # Initialize columns and rows with default empty values
        columns = []
        rows = []
        
        if db_type == 'postgres':
            if not enterprise.postgres_host or not enterprise.postgres_admin_user or not enterprise.postgres_admin_password:
                return jsonify({'error': 'PostgreSQL configuration not found'}), 400
            if is_shared:
                db_uri = f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}" \
                        f"@{enterprise.postgres_host}:{enterprise.postgres_port}/{selected_db}"
            else:
                user = User.query.filter_by(username=session['username']).first()
                if not user or not user.postgres_password:
                    return jsonify({'error': 'PostgreSQL credentials not found for user'}), 401
                db_uri = f"postgresql://{session['username']}:{user.postgres_password}" \
                        f"@{enterprise.postgres_host}:{enterprise.postgres_port}/{selected_db}"
            
            engine = create_engine(db_uri)
            with engine.connect() as conn:
                escaped_table = table_name.replace("'", "''")
                query = text(f'SELECT * FROM "{escaped_table}" LIMIT 10')
                result = conn.execute(query)
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]

        elif db_type == 'mysql':
            if not enterprise.mysql_host or not enterprise.mysql_admin_user or not enterprise.mysql_admin_password:
                return jsonify({'error': 'MySQL configuration not found'}), 400
            if is_shared:
                db_uri = f"mysql+pymysql://{enterprise.mysql_admin_user}:{enterprise.mysql_admin_password}" \
                        f"@{enterprise.mysql_host}:{enterprise.mysql_port}/{selected_db}"
            else:
                user = User.query.filter_by(username=session['username']).first()
                if not user or not user.mysql_password:
                    return jsonify({'error': 'MySQL credentials not found for user'}), 401
                db_uri = f"mysql+pymysql://{session['username']}:{user.mysql_password}" \
                        f"@{enterprise.mysql_host}:{enterprise.mysql_port}/{selected_db}"
            
            engine = create_engine(db_uri)
            with engine.connect() as conn:
                escaped_table = table_name.replace("'", "''")
                query = text(f'SELECT * FROM `{escaped_table}` LIMIT 10')
                result = conn.execute(query)
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]

        elif db_type == 'sqlserver':
            if not all([enterprise.sqlserver_host, enterprise.sqlserver_admin_user, enterprise.sqlserver_admin_password]):
                return jsonify({'error': 'SQL Server not configured'}), 400

            # Handle both instance and port-based connections
            if enterprise.sqlserver_instance:
                server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}"
            else:
                server = f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}" if enterprise.sqlserver_port else enterprise.sqlserver_host

            if is_shared:
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"UID={enterprise.sqlserver_admin_user};"
                    f"PWD={enterprise.sqlserver_admin_password};"
                    f"DATABASE={selected_db}"
                )
            else:
                user = User.query.filter_by(username=session['username']).first()
                if not user or not user.sqlserver_password:
                    return jsonify({'error': 'SQL Server credentials not found for user'}), 401
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={server};"
                    f"UID={session['username']};"
                    f"PWD={user.sqlserver_password};"
                    f"DATABASE={selected_db}"
                )
            
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            query = f"SELECT TOP 10 * FROM [{table_name}]"
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            conn.close()

        return jsonify({
            'columns': list(columns) if columns else [],
            'rows': rows if rows else []
        })

    except Exception as e:
        logger.error(f"Error fetching preview for table {table_name} in database {selected_db}: {str(e)}")
        return jsonify({
            'error': str(e),
            'columns': [],
            'rows': []
        }), 500
    
@app.route('/admin/users')
@login_required
def admin_users():
    if not session.get('is_admin'):
        abort(403)
    
    users = User.query.all()  # Uses SQLite automatically
    return render_template('admin/users.html', users=users)

@app.route('/admin/logs')
@login_required
def admin_logs():
    logs = AdminLog.query.order_by(AdminLog.timestamp.desc()).limit(100).all()
    return render_template('admin/logs.html', logs=logs)

@app.route('/api/client-databases')
@login_required
def client_databases():
    try:
        with client_engine.connect() as conn:
            result = conn.execute(text("SHOW DATABASES"))
            databases = [row[0] for row in result if row[0] not in 
                        {'information_schema', 'mysql', 'performance_schema', 'sys'}]
            return jsonify({'databases': databases})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def get_admin_db():
    """Get a connection to the admin SQLite database"""
    return sqlite3.connect(Path(app.instance_path) / 'admin_data.db')

def backup_admin_db():
    """Create a backup of the admin database"""
    backup_path = Path(app.config['UPLOAD_FOLDER']) / 'backups' / f'admin_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    
    with get_admin_db() as src:
        with sqlite3.connect(backup_path) as dst:
            src.backup(dst)
    return backup_path

@app.route('/admin/backup', methods=['POST'])
@login_required
def admin_backup():
    if not session.get('is_admin'):
        abort(403)
    
    try:
        backup_file = backup_admin_db()
        return send_file(
            backup_file,
            as_attachment=True,
            download_name=backup_file.name
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/restore', methods=['POST'])
@login_required
def admin_restore():
    if not session.get('is_admin'):
        abort(403)
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        temp_path = Path(app.config['UPLOAD_FOLDER']) / 'temp_restore.db'
        file.save(temp_path)
        
        # Close existing connections
        db.session.remove()
        
        # Restore backup
        with sqlite3.connect(temp_path) as src:
            with get_admin_db() as dst:
                src.backup(dst)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_path.exists():
            temp_path.unlink()
            
@app.route('/payment', methods=['GET', 'POST'])
def payment():
    if request.method == 'POST':
        # Handle payment processing here
        # This is a placeholder - implement actual payment processing with a payment gateway
        try:
            # Example: Process payment with card details
            card_number = request.form.get('card-number')
            cardholder_name = request.form.get('cardholder-name')
            expiry_date = request.form.get('expiry-date')
            cvv = request.form.get('cvv')
            country = request.form.get('country')
            plan_name = request.form.get('plan-name')
            plan_price = request.form.get('plan-price')
            user_count = request.form.get('user-count')
            plan_type = request.form.get('plan-type')

            # Log the payment attempt
            log_admin_action(
                session['username'],
                "Payment attempt",
                f"Attempted subscription to {plan_name} plan for {user_count} users ({plan_type})",
                enterprise_id=session['enterprise_id']
            )

            # Here you would integrate with a payment gateway like Stripe
            # For demo purposes, we'll just log and show success
            flash(f"Payment successful for {plan_name} plan!", 'success')
            return redirect(url_for('app_page'))

        except Exception as e:
            logger.error(f"Payment processing error: {str(e)}")
            flash(f"Payment failed: {str(e)}", 'error')
            return redirect(url_for('payment'))

    # GET request - render payment page with plan details
    plan_name = request.args.get('plan')
    plan_price = request.args.get('price')
    user_count = request.args.get('users')
    plan_type = request.args.get('type')

    return render_template('payment.html',
                         plan_name=plan_name,
                         plan_price=plan_price,
                         user_count=user_count,
                         plan_type=plan_type)  
    
@app.route('/api/save-plan', methods=['POST'])
def save_plan():
    try:
        data = request.form
        email = data.get('email')
        plan_name = data.get('plan_name')
        user_count = data.get('user_count')
        price = data.get('price')
        plan_type = data.get('plan_type')
        payment_status = data.get('payment_status')

        if not all([email, plan_name, user_count, price, plan_type, payment_status]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        new_plan = Plan(
            email=email,
            plan_name=plan_name,
            user_count=user_count,
            price=price,
            plan_type=plan_type,
            payment_status=payment_status
        )
        db.session.add(new_plan)
        db.session.commit()

        flash('Plan saved successfully! Please proceed to enterprise setup.', 'success')
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500         
        
        
def create_postgres_user_and_db(username, password):
    try:
        enterprise = get_current_enterprise()
        if not enterprise or not enterprise.postgres_host:
            raise ValueError("PostgreSQL configuration not found. Please configure PostgreSQL connection first.")

        # Create connection string
        conn_str = f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}@{enterprise.postgres_host}:{enterprise.postgres_port}/postgres"
        
        # Create engine with autocommit enabled
        engine = create_engine(conn_str, isolation_level="AUTOCOMMIT")
        
        db_name = sanitize_db_name(username)
        
        # Check if database already exists
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :db_name"), {"db_name": db_name})
            if result.fetchone():
                raise ValueError(f"Database {db_name} already exists")

        # Create the database
        with engine.connect() as conn:
            conn.execute(text(f"CREATE DATABASE {db_name}"))
        
        # Create user and grant privileges
        with engine.connect() as conn:
            escaped_username = username.replace("'", "''")
            escaped_password = password.replace("'", "''")
            conn.execute(text(f"CREATE USER {escaped_username} WITH PASSWORD '{escaped_password}'"))
            
            # Connect to the new database to grant privileges
            db_conn_str = f"postgresql://{enterprise.postgres_admin_user}:{enterprise.postgres_admin_password}@{enterprise.postgres_host}:{enterprise.postgres_port}/{db_name}"
            db_engine = create_engine(db_conn_str, isolation_level="AUTOCOMMIT")
            with db_engine.connect() as db_conn:
                # Grant database-level privileges
                db_conn.execute(text(f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO {escaped_username}"))
                # Grant SELECT, INSERT, UPDATE, DELETE on all tables in public schema
                db_conn.execute(text(f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {escaped_username}"))
                # Grant USAGE on the public schema to allow access to tables
                db_conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {escaped_username}"))
                # Ensure privileges are applied to future tables
                db_conn.execute(text(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {escaped_username}"))
            
        logger.info(f"Created PostgreSQL user {username} and database {db_name} with full table privileges")
        return db_name, password

    except Exception as e:
        logger.error(f"Error creating PostgreSQL user or database for {username}: {str(e)}")
        # Attempt cleanup
        try:
            cleanup_engine = create_engine(conn_str, isolation_level="AUTOCOMMIT")
            with cleanup_engine.connect() as cleanup_conn:
                cleanup_conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
                cleanup_conn.execute(text(f"DROP USER IF EXISTS {escaped_username}"))
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed after error: {str(cleanup_error)}")
        
        raise ValueError(f"Failed to create PostgreSQL user and database: {str(e)}")

@app.route('/api/test-postgres-connection', methods=['POST'])
def test_postgres_connection():
    try:
        data = request.get_json()
        connection_string = f"postgresql://{data['user']}:{data['password']}@{data['host']}:{data['port']}/postgres"
        
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/save-postgres-config', methods=['POST'])
@require_enterprise_auth
def api_save_postgres_config():
    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'success': False, 'error': 'Enterprise not found'}), 404

        data = request.get_json()
        
        # Validate required fields
        required_fields = ['host', 'port', 'user', 'password']
        if not all(field in data for field in required_fields):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        # Test the connection
        connection_string = f"postgresql://{data['user']}:{data['password']}@{data['host']}:{data['port']}/postgres"
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            return jsonify({'success': False, 'error': f'Connection test failed: {str(e)}'}), 400

        # Update enterprise with PostgreSQL config
        enterprise.postgres_host = data['host']
        enterprise.postgres_port = data['port']
        enterprise.postgres_admin_user = data['user']
        enterprise.postgres_admin_password = data['password']
        db.session.commit()

        # Save to file for backward compatibility
        with open(POSTGRES_CONFIG_FILE, 'w') as f:
            json.dump(data, f)
        
        # Update app config
        app.config['POSTGRES_HOST'] = data['host']
        app.config['POSTGRES_PORT'] = data['port']
        app.config['POSTGRES_ADMIN_USER'] = data['user']
        app.config['POSTGRES_ADMIN_PASSWORD'] = data['password']
        
        # Get all non-system PostgreSQL databases with sizes
        databases = []
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false"))
                system_dbs = {'postgres', 'template0', 'template1'}
                for row in result:
                    db_name = row[0]
                    if db_name not in system_dbs:
                        size = get_database_size(conn, db_name, 'postgres')
                        databases.append({
                            'name': db_name,
                            'type': 'postgres',
                            'size': size if size is not None else 'unknown'
                        })
            logger.info(f"Fetched PostgreSQL databases: {databases}")
        except Exception as e:
            logger.error(f"Error fetching PostgreSQL databases: {str(e)}")
            return jsonify({'success': False, 'error': f'Failed to fetch databases: {str(e)}'}), 500

        # Get current shared databases
        shared_dbs = [
            {'name': db.database_name, 'type': db.database_type}
            for db in SharedDatabase.query.filter_by(enterprise_id=enterprise.id).all()
        ]

        return jsonify({
            'success': True,
            'databases': databases,
            'shared_databases': shared_dbs,
            'message': 'PostgreSQL configuration saved successfully'
        })
            
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving PostgreSQL config: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error saving configuration: {str(e)}'
        }), 500

@app.route('/api/get-postgres-config', methods=['GET'])
def get_postgres_config():
    try:
        enterprise = get_current_enterprise()
        if enterprise and enterprise.postgres_host:
            return jsonify({
                'success': True,
                'config': {
                    'host': enterprise.postgres_host,
                    'port': enterprise.postgres_port,
                    'user': enterprise.postgres_admin_user,
                    # Never return password
                }
            })
        return jsonify({
            'success': False,
            'message': 'No PostgreSQL configuration found'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/check-postgres-config', methods=['GET'])
def check_postgres_config():
    enterprise = get_current_enterprise()
    if enterprise and enterprise.postgres_host:
        return jsonify({
            'has_config': True,
            'config': {
                'host': enterprise.postgres_host,
                'port': enterprise.postgres_port,
                'user': enterprise.postgres_admin_user,
                # Never return password
            }
        })
    return jsonify({
        'has_config': False,
        'config': {
            'host': '',
            'port': '',
            'user': '',
        }
    }) 

# New function to create SQL Server user and database
def create_sqlserver_user_and_db(username, password):
    try:
        enterprise = get_current_enterprise()
        if not enterprise or not enterprise.sqlserver_host:
            raise ValueError("SQL Server configuration not found. Please configure SQL Server connection first.")

        # Create server string based on whether instance name is provided
        server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
        
        # Create connection string for master database
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={server};"
            f"UID={enterprise.sqlserver_admin_user};"
            f"PWD={enterprise.sqlserver_admin_password};"
            f"DATABASE=master;"
            f"Connection Timeout=30;"
        )
        
        db_name = sanitize_db_name(username)
        
        # Connect to SQL Server
        conn = pyodbc.connect(conn_str)
        conn.autocommit = True  # Important for DDL statements
        cursor = conn.cursor()
        
        try:
            # Check if login already exists
            cursor.execute("SELECT name FROM sys.server_principals WHERE name = ?", username)
            if cursor.fetchone():
                logger.info(f"SQL Server login {username} already exists")
            else:
                # Create login - use direct string formatting for password since it's a DDL statement
                cursor.execute(f"CREATE LOGIN [{username}] WITH PASSWORD = '{password}', CHECK_POLICY=OFF, CHECK_EXPIRATION=OFF")
                logger.info(f"Created SQL Server login {username}")
            
            # Check if database exists
            cursor.execute("SELECT name FROM sys.databases WHERE name = ?", db_name)
            if cursor.fetchone():
                logger.info(f"Database {db_name} already exists")
            else:
                # Create database
                cursor.execute(f"CREATE DATABASE [{db_name}]")
                logger.info(f"Created database {db_name}")
            
            # Connect to the new database to create user
            db_conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={server};"
                f"UID={enterprise.sqlserver_admin_user};"
                f"PWD={enterprise.sqlserver_admin_password};"
                f"DATABASE={db_name};"
            )
            db_conn = pyodbc.connect(db_conn_str)
            db_conn.autocommit = True
            db_cursor = db_conn.cursor()
            
            try:
                # Create user in the database and grant permissions
                db_cursor.execute(f"CREATE USER [{username}] FOR LOGIN [{username}]")
                db_cursor.execute(f"ALTER ROLE db_owner ADD MEMBER [{username}]")
                logger.info(f"Granted db_owner role to {username} on database {db_name}")
                
                # Grant access to any shared SQL Server databases
                shared_dbs = SharedDatabase.query.filter_by(
                    enterprise_id=enterprise.id,
                    database_type='sqlserver'
                ).all()
                
                for shared_db in shared_dbs:
                    try:
                        # Connect to the shared database
                        shared_db_conn_str = (
                            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                            f"SERVER={server};"
                            f"UID={enterprise.sqlserver_admin_user};"
                            f"PWD={enterprise.sqlserver_admin_password};"
                            f"DATABASE={shared_db.database_name};"
                        )
                        shared_db_conn = pyodbc.connect(shared_db_conn_str)
                        shared_db_conn.autocommit = True
                        shared_db_cursor = shared_db_conn.cursor()
                        
                        # Create user and grant SELECT permission
                        shared_db_cursor.execute(f"""
                            IF NOT EXISTS (SELECT * FROM sys.database_principals WHERE name = '{username}')
                            CREATE USER [{username}] FOR LOGIN [{username}];
                            GRANT SELECT TO [{username}];
                        """)
                        logger.info(f"Granted SELECT access to {username} on shared database {shared_db.database_name}")
                        
                        shared_db_cursor.close()
                        shared_db_conn.close()
                    except Exception as e:
                        logger.error(f"Error granting access to shared database {shared_db.database_name}: {str(e)}")
                        continue
            finally:
                db_cursor.close()
                db_conn.close()
            
            return db_name, password
            
        finally:
            cursor.close()
            conn.close()

    except Exception as e:
        logger.error(f"Error creating SQL Server user or database for {username}: {str(e)}")
        # Attempt cleanup
        try:
            if 'conn_str' in locals():
                cleanup_conn = pyodbc.connect(conn_str)
                cleanup_conn.autocommit = True  # Important for DROP statements
                cleanup_cursor = cleanup_conn.cursor()
                
                # Drop database if exists
                cleanup_cursor.execute(f"IF EXISTS (SELECT name FROM sys.databases WHERE name = '{db_name}') DROP DATABASE [{db_name}]")
                
                # Drop login if exists
                cleanup_cursor.execute(f"IF EXISTS (SELECT name FROM sys.server_principals WHERE name = '{username}') DROP LOGIN [{username}]")
                
                cleanup_cursor.close()
                cleanup_conn.close()
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed after error: {str(cleanup_error)}")
        
        raise ValueError(f"Failed to create SQL Server user and database: {str(e)}")
    
# New route for testing SQL Server connection
@app.route('/api/test-sqlserver-connection', methods=['POST'])
def test_sqlserver_connection():
    try:
        data = request.get_json()
        
        # Build connection string based on whether instance name is provided
        if data.get('instance'):
            # For named instances
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={data['host']}\\{data['instance']};"
                f"UID={data['user']};"
                f"PWD={data['password']};"
                f"Database=master;"
                f"Connection Timeout=30;"
            )
        else:
            # For default instance with port
            conn_str = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={data['host']},{data['port']};"
                f"UID={data['user']};"
                f"PWD={data['password']};"
                f"Database=master;"
                f"Connection Timeout=30;"
            )
        
        try:
            conn = pyodbc.connect(conn_str)
            conn.close()
            return jsonify({'success': True})
        except pyodbc.Error as e:
            error_msg = str(e)
            if "Named Pipes Provider" in error_msg:
                error_msg += "\n\nPossible solutions:\n1. Enable TCP/IP protocol in SQL Server Configuration Manager\n2. Ensure SQL Server Browser service is running\n3. Check firewall settings"
            return jsonify({
                'success': False, 
                'error': error_msg
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
# New route for saving SQL Server configuration
@app.route('/api/save-sqlserver-config', methods=['POST'])
@require_enterprise_auth
def api_save_sqlserver_config():
    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'success': False, 'error': 'Enterprise not found'}), 404

        data = request.get_json()
        
        # Validate required fields
        required_fields = ['host', 'user', 'password']
        if not all(field in data for field in required_fields):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        # If instance name is provided, we don't need port
        if data.get('instance'):
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={data['host']}\\{data['instance']};"
                f"UID={data['user']};"
                f"PWD={data['password']};"
                f"DATABASE=master"
            )
        else:
            # For default instance, require port
            if not data.get('port'):
                return jsonify({'success': False, 'error': 'Port is required for default instance'}), 400
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={data['host']},{data['port']};"
                f"UID={data['user']};"
                f"PWD={data['password']};"
                f"DATABASE=master"
            )

        # Test the connection
        try:
            conn = pyodbc.connect(connection_string)
            conn.close()
        except Exception as e:
            logger.error(f"Failed to connect to SQL Server: {str(e)}")
            return jsonify({'success': False, 'error': f'Connection test failed: {str(e)}'}), 400

        # Update enterprise with SQL Server config
        enterprise.sqlserver_host = data['host']
        enterprise.sqlserver_instance = data.get('instance', '')
        enterprise.sqlserver_port = data.get('port', 1433)
        enterprise.sqlserver_admin_user = data['user']
        enterprise.sqlserver_admin_password = data['password']
        db.session.commit()

        # Save to file for backward compatibility
        with open(SQLSERVER_CONFIG_FILE, 'w') as f:
            json.dump({
                'host': data['host'],
                'instance': data.get('instance', ''),
                'port': data.get('port', 1433),
                'user': data['user'],
                'password': data['password']
            }, f)
        
        # Get all non-system SQL Server databases with sizes
        databases = []
        try:
            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sys.databases WHERE name NOT IN ('master', 'tempdb', 'model', 'msdb')")
            for row in cursor:
                db_name = row.name
                size = get_database_size(cursor, db_name, 'sqlserver')
                databases.append({
                    'name': db_name,
                    'type': 'sqlserver',
                    'size': size if size is not None else 'unknown'
                })
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error fetching SQL Server databases: {str(e)}")

        return jsonify({
            'success': True,
            'databases': databases,
            'message': 'SQL Server configuration saved successfully'
        })
            
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving SQL Server config: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error saving configuration: {str(e)}'
        }), 500

# New route for getting SQL Server configuration
@app.route('/api/get-sqlserver-config', methods=['GET'])
def get_sqlserver_config():
    try:
        enterprise = get_current_enterprise()
        if enterprise and enterprise.sqlserver_host:
            return jsonify({
                'success': True,
                'config': {
                    'host': enterprise.sqlserver_host,
                    'port': enterprise.sqlserver_port,
                    'user': enterprise.sqlserver_admin_user,
                    # Never return password
                }
            })
        return jsonify({
            'success': False,
            'message': 'No SQL Server configuration found'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# New route for checking SQL Server configuration
@app.route('/api/check-sqlserver-config', methods=['GET'])
def check_sqlserver_config():
    enterprise = get_current_enterprise()
    if enterprise and enterprise.sqlserver_host:
        return jsonify({
            'has_config': True,
            'config': {
                'host': enterprise.sqlserver_host,
                'port': enterprise.sqlserver_port,
                'user': enterprise.sqlserver_admin_user,
                # Never return password
            }
        })
    return jsonify({
        'has_config': False,
        'config': {
            'host': '',
            'port': '',
            'user': '',
        }
    })
  
# code for dictionary  
@app.route('/api/upload-dictionary', methods=['POST'])
@require_enterprise_auth
def upload_dictionary():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    database = request.form.get('database')
    table = request.form.get('table')
    db_type = request.form.get('type', 'mysql')

    if not database or not table:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'error': 'Enterprise not found'}), 404

        # Process the uploaded file
        dictionary = None
        filename = file.filename.lower()
        
        if filename.endswith('.json'):
            try:
                # Read the file content first
                file_content = file.stream.read().decode('utf-8')
                dictionary = json.loads(file_content)
            except json.JSONDecodeError as e:
                return jsonify({'error': f'Invalid JSON file: {str(e)}'}), 400
            except Exception as e:
                return jsonify({'error': f'Error reading JSON file: {str(e)}'}), 400
        elif filename.endswith('.csv'):
            try:
                import pandas as pd
                from io import StringIO
                
                # Read CSV into DataFrame
                file_content = file.stream.read().decode('utf-8')
                df = pd.read_csv(StringIO(file_content))
                
                # Convert DataFrame to dictionary format
                dictionary = {}
                for _, row in df.iterrows():
                    # Handle case where CSV might have headers or not
                    if len(row) >= 2:
                        column_name = str(row.iloc[0]).strip()
                        description = str(row.iloc[1]).strip() if len(row) > 1 else ""
                        data_type = str(row.iloc[2]).strip() if len(row) > 2 else ""
                        
                        dictionary[column_name] = {
                            'description': description,
                            'type': data_type
                        }
                    else:
                        # If only one column, use it as both name and description
                        column_name = str(row.iloc[0]).strip()
                        dictionary[column_name] = {
                            'description': column_name,
                            'type': 'unknown'
                        }
            except Exception as e:
                return jsonify({'error': f'Error processing CSV: {str(e)}'}), 400
        else:
            return jsonify({'error': 'Unsupported file type. Please upload JSON or CSV.'}), 400

        # Validate the dictionary structure
        if not isinstance(dictionary, dict):
            return jsonify({'error': 'Invalid dictionary format. Must be a key-value mapping.'}), 400

        # Check if dictionary already exists
        existing_dict = DataDictionary.query.filter_by(
            table_name=table,
            database_name=database,
            database_type=db_type,
            enterprise_id=enterprise.id
        ).first()

        if existing_dict:
            existing_dict.dictionary = dictionary
            existing_dict.updated_at = datetime.utcnow()
        else:
            new_dict = DataDictionary(
                table_name=table,
                database_name=database,
                database_type=db_type,
                dictionary=dictionary,
                enterprise_id=enterprise.id
            )
            db.session.add(new_dict)

        db.session.commit()
        
        log_admin_action(
            session['username'],
            "Uploaded data dictionary",
            f"For table {table} in database {database}",
            enterprise_id=enterprise.id
        )
        
        return jsonify({'success': True, 'message': 'Dictionary saved successfully'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving dictionary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-dictionary', methods=['GET'])
@require_enterprise_auth
def get_dictionary():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    database = request.args.get('database')
    table = request.args.get('table')
    db_type = request.args.get('type', 'mysql')

    if not database or not table:
        return jsonify({'error': 'Database and table name are required'}), 400

    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'error': 'Enterprise not found'}), 404

        dictionary = DataDictionary.query.filter_by(
            table_name=table,
            database_name=database,
            database_type=db_type,
            enterprise_id=enterprise.id
        ).first()

        if dictionary:
            return jsonify({
                'success': True,
                'dictionary': dictionary.dictionary
            })
        else:
            return jsonify({
                'success': True,
                'dictionary': None
            })
    except Exception as e:
        logger.error(f"Error getting dictionary: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-dictionary', methods=['POST'])
@require_enterprise_auth
def delete_dictionary():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    data = request.get_json()
    if not data or 'database' not in data or 'table' not in data:
        return jsonify({'error': 'Missing required fields'}), 400

    try:
        enterprise = get_current_enterprise()
        if not enterprise:
            return jsonify({'error': 'Enterprise not found'}), 404

        dictionary = DataDictionary.query.filter_by(
            table_name=data['table'],
            database_name=data['database'],
            database_type=data.get('type', 'mysql'),
            enterprise_id=enterprise.id
        ).first()

        if dictionary:
            db.session.delete(dictionary)
            db.session.commit()
            
            log_admin_action(
                session['username'],
                "Deleted data dictionary",
                f"For table {data['table']} in database {data['database']}",
                enterprise_id=enterprise.id
            )
            
            return jsonify({'success': True, 'message': 'Dictionary deleted successfully'})
        else:
            return jsonify({'error': 'Dictionary not found'}), 404
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting dictionary: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)