import os
from dotenv import load_dotenv
from pathlib import Path
import json

load_dotenv()

# Base directory
basedir = Path(__file__).parent

class Config:
    # Admin SQLite Database Configuration
    SQLALCHEMY_DATABASE_URI = f'sqlite:///{basedir / "instance" / "admin_data.db"}'
    
    # MySQL Admin Configuration (for client operations)
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
    MYSQL_ADMIN_USER = os.getenv('MYSQL_ADMIN_USER', 'root')
    MYSQL_ADMIN_PASSWORD = os.getenv('MYSQL_ADMIN_PASSWORD', 'admin')
    
    # Compute mysql_client_uri as a string
    MYSQL_CLIENT_URI = f"mysql+pymysql://{MYSQL_ADMIN_USER}:{MYSQL_ADMIN_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/"
    
    # Define SQLALCHEMY_BINDS with the computed URI
    SQLALCHEMY_BINDS = {
        'admin': SQLALCHEMY_DATABASE_URI,
        'client': MYSQL_CLIENT_URI
    }
    
    # File Storage Configuration
    REPORT_FOLDER = str(basedir / 'reports')
    UPLOAD_FOLDER = str(basedir / 'Uploads')
    ENTERPRISE_FOLDER = str(basedir / 'Uploads' / 'enterprises')
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    
    # Security Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # PostgreSQL Configuration
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_ADMIN_USER = os.getenv('POSTGRES_ADMIN_USER', 'postgres')
    POSTGRES_ADMIN_PASSWORD = os.getenv('POSTGRES_ADMIN_PASSWORD', 'admin')
    POSTGRES_URI = f"postgresql://{POSTGRES_ADMIN_USER}:{POSTGRES_ADMIN_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}"
    
    # SQL Server Configuration
    SQLSERVER_HOST = os.getenv('SQLSERVER_HOST', 'localhost')
    SQLSERVER_INSTANCE = os.getenv('SQLSERVER_INSTANCE', '')  # Add this line for named instance
    SQLSERVER_PORT = os.getenv('SQLSERVER_PORT', '1433')
    SQLSERVER_ADMIN_USER = os.getenv('SQLSERVER_ADMIN_USER', 'sa')
    SQLSERVER_ADMIN_PASSWORD = os.getenv('SQLSERVER_ADMIN_PASSWORD', 'your_strong_password')
    
    # Updated connection string that handles both default and named instances
    if SQLSERVER_INSTANCE:
        SQLSERVER_URI = f"mssql+pyodbc://{SQLSERVER_ADMIN_USER}:{SQLSERVER_ADMIN_PASSWORD}@{SQLSERVER_HOST}\\{SQLSERVER_INSTANCE}?driver=ODBC+Driver+17+for+SQL+Server"
    else:
        SQLSERVER_URI = f"mssql+pyodbc://{SQLSERVER_ADMIN_USER}:{SQLSERVER_ADMIN_PASSWORD}@{SQLSERVER_HOST}:{SQLSERVER_PORT}?driver=ODBC+Driver+17+for+SQL+Server"
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_API_VERSION_GPT_4 = os.getenv("AZURE_API_VERSION_GPT_4")
    AZURE_OPENAI_GPT_4_TURBO_MODEL = os.getenv("AZURE_OPENAI_GPT_4_TURBO_MODEL")
    AZURE_OPENAI_EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL")

    @staticmethod
    def init_app(app):
        """Initialize app-specific configurations and ensure directories exist."""
        for folder in [app.config['REPORT_FOLDER'], 
                      app.config['UPLOAD_FOLDER'],
                      app.config['ENTERPRISE_FOLDER']]:
            os.makedirs(folder, exist_ok=True)

        # Load MySQL config from file if it exists
        mysql_config_file = os.path.join(app.config['UPLOAD_FOLDER'], 'mysql_config.json')
        if os.path.exists(mysql_config_file):
            with open(mysql_config_file, 'r') as f:
                config = json.load(f)
                app.config['MYSQL_HOST'] = config.get('host', app.config['MYSQL_HOST'])
                app.config['MYSQL_PORT'] = config.get('port', app.config['MYSQL_PORT'])
                app.config['MYSQL_ADMIN_USER'] = config.get('user', app.config['MYSQL_ADMIN_USER'])
                app.config['MYSQL_ADMIN_PASSWORD'] = config.get('password', app.config['MYSQL_ADMIN_PASSWORD'])
                # Update SQLALCHEMY_BINDS['client'] with the loaded MySQL configuration
                app.config['SQLALCHEMY_BINDS']['client'] = (
                    f"mysql+pymysql://{app.config['MYSQL_ADMIN_USER']}:"
                    f"{app.config['MYSQL_ADMIN_PASSWORD']}@{app.config['MYSQL_HOST']}:"
                    f"{app.config['MYSQL_PORT']}/"
                )

        # Load PostgreSQL config from file if it exists
        postgres_config_file = os.path.join(app.config['UPLOAD_FOLDER'], 'postgres_config.json')
        if os.path.exists(postgres_config_file):
            with open(postgres_config_file, 'r') as f:
                config = json.load(f)
                app.config['POSTGRES_HOST'] = config.get('host')
                app.config['POSTGRES_PORT'] = config.get('port', '5432')
                app.config['POSTGRES_ADMIN_USER'] = config.get('user')
                app.config['POSTGRES_ADMIN_PASSWORD'] = config.get('password')
                app.config['POSTGRES_URI'] = (
                    f"postgresql://{app.config['POSTGRES_ADMIN_USER']}:"
                    f"{app.config['POSTGRES_ADMIN_PASSWORD']}@{app.config['POSTGRES_HOST']}:"
                    f"{app.config['POSTGRES_PORT']}"
                )
                
        # Load SQL Server config from file if it exists
        sqlserver_config_file = os.path.join(app.config['UPLOAD_FOLDER'], 'sqlserver_config.json')
        if os.path.exists(sqlserver_config_file):
            with open(sqlserver_config_file, 'r') as f:
                config = json.load(f)
                app.config['SQLSERVER_HOST'] = config.get('host')
                app.config['SQLSERVER_PORT'] = config.get('port', '1433')
                app.config['SQLSERVER_ADMIN_USER'] = config.get('user')
                app.config['SQLSERVER_ADMIN_PASSWORD'] = config.get('password')
                app.config['SQLSERVER_URI'] = (
                    f"mssql+pyodbc://{app.config['SQLSERVER_ADMIN_USER']}:"
                    f"{app.config['SQLSERVER_ADMIN_PASSWORD']}@{app.config['SQLSERVER_HOST']}:"
                    f"{app.config['SQLSERVER_PORT']}/?driver=ODBC+Driver+17+for+SQL+Server"
                )