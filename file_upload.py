from flask import Blueprint, current_app, session, request, jsonify, render_template
import pandas as pd
import mysql.connector
import psycopg2
import pyodbc
from werkzeug.utils import secure_filename
import os
from models import db, User, Enterprise
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define Blueprint for file upload routes
file_upload_bp = Blueprint('file_upload', __name__)

def sanitize_db_name(db_name):
    """Sanitize database and table names to comply with database naming conventions."""
    if not db_name or not isinstance(db_name, str):
        db_name = "table_unknown"
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', db_name.strip())
    if not sanitized[0].isalpha() and not sanitized[0] == '_':
        sanitized = 'tbl_' + sanitized
    # Ensure the name is not empty and within SQL Server's max identifier length
    sanitized = sanitized[:128] if sanitized else 'table_unknown'
    return sanitized

def sanitize_column_name(col_name):
    """Sanitize column name to comply with database naming conventions."""
    if not col_name or not isinstance(col_name, str):
        col_name = "column_unknown"
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', col_name.strip())
    if not sanitized[0].isalpha() and not sanitized[0] == '_':
        sanitized = 'col_' + sanitized
    # SQL Server reserved words (expanded list)
    reserved_words = {
        'select', 'insert', 'update', 'delete', 'from', 'where', 'table', 'database',
        'float', 'int', 'varchar', 'nvarchar', 'bit', 'datetime', 'create', 'drop',
        'alter', 'index', 'view', 'procedure', 'trigger', 'constraint'
    }
    if sanitized.lower() in reserved_words:
        sanitized += '_col'
    # Ensure the name is not empty and within SQL Server's max identifier length
    sanitized = sanitized[:128] if sanitized else 'column_unknown'
    return sanitized

def create_table_from_file(file, db_name, table_name, db_type):
    """Create a table in the specified database from the uploaded CSV/Excel file."""
    temp_path = None
    try:
        # Get current user from session
        username = session.get('username')
        if not username:
            raise Exception("User not authenticated")

        # Fetch user and enterprise from database
        user = User.query.filter_by(username=username).first()
        if not user:
            raise Exception("User not found in database")

        enterprise = Enterprise.query.filter_by(id=user.enterprise_id).first()
        if not enterprise:
            raise Exception("Enterprise not found")

        # Get database configuration
        db_config = {}
        if db_type == 'mysql':
            db_config = {
                'host': enterprise.mysql_host,
                'port': enterprise.mysql_port,
                'user': username,
                'password': user.mysql_password,
                'database': db_name
            }
        elif db_type == 'postgres':
            db_config = {
                'host': enterprise.postgres_host,
                'port': enterprise.postgres_port,
                'user': username,
                'password': user.postgres_password,
                'database': db_name
            }
        elif db_type == 'sqlserver':
            server = f"{enterprise.sqlserver_host}\\{enterprise.sqlserver_instance}" if enterprise.sqlserver_instance else f"{enterprise.sqlserver_host},{enterprise.sqlserver_port}"
            db_config = {
                'driver': '{ODBC Driver 17 for SQL Server}',
                'server': server,
                'uid': username,
                'pwd': user.sqlserver_password,
                'database': db_name
            }
        else:
            raise Exception("Invalid database type")

        if not db_config.get('password') and db_type != 'sqlserver':
            raise Exception(f"{db_type.upper()} credentials not found for user")
        if db_type == 'sqlserver' and not db_config.get('pwd'):
            raise Exception("SQL Server credentials not found for user")

        # Save and process the uploaded file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        if filename.endswith('.csv'):
            df = pd.read_csv(temp_path, encoding='utf-8')
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(temp_path)
        else:
            raise Exception("Unsupported file format. Please upload a CSV or Excel file.")

        if df.empty:
            raise Exception("The uploaded file is empty.")

        table_name = sanitize_db_name(table_name)
        
        # Sanitize column names
        sanitized_columns = []
        seen_columns = {}
        for col in df.columns:
            sanitized_col = sanitize_column_name(str(col))
            if sanitized_col in seen_columns:
                count = seen_columns[sanitized_col] + 1
                seen_columns[sanitized_col] = count
                sanitized_col = f"{sanitized_col}_{count}"
            else:
                seen_columns[sanitized_col] = 0
            sanitized_columns.append(sanitized_col)
        
        df.columns = sanitized_columns

        # Generate column definitions
        columns = []
        for col, dtype in zip(df.columns, df.dtypes):
            if db_type == 'mysql':
                if pd.api.types.is_integer_dtype(dtype):
                    col_type = 'INT'
                elif pd.api.types.is_float_dtype(dtype):
                    col_type = 'FLOAT'
                elif pd.api.types.is_bool_dtype(dtype):
                    col_type = 'BOOLEAN'
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    col_type = 'DATETIME'
                else:
                    sample = df[col].dropna().astype(str).str.len().max()
                    max_length = min(max(int(sample), 255), 65535) if pd.notnull(sample) else 255
                    col_type = f'VARCHAR({max_length})'
                columns.append(f"`{col.replace('`', '``')}` {col_type}")
            elif db_type == 'postgres':
                if pd.api.types.is_integer_dtype(dtype):
                    col_type = 'INTEGER'
                elif pd.api.types.is_float_dtype(dtype):
                    col_type = 'DOUBLE PRECISION'
                elif pd.api.types.is_bool_dtype(dtype):
                    col_type = 'BOOLEAN'
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    col_type = 'TIMESTAMP'
                else:
                    sample = df[col].dropna().astype(str).str.len().max()
                    max_length = min(max(int(sample), 255), 10485760) if pd.notnull(sample) else 255
                    col_type = f'VARCHAR({max_length})'
                
                # This is the fixed line - no nested quotes in the f-string
                escaped_col = col.replace('"', '""')
                columns.append(f'"{escaped_col}" {col_type}')
            elif db_type == 'sqlserver':
                if pd.api.types.is_integer_dtype(dtype):
                    col_type = 'INT'
                elif pd.api.types.is_float_dtype(dtype):
                    col_type = 'FLOAT(53)'
                elif pd.api.types.is_bool_dtype(dtype):
                    col_type = 'BIT'
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    col_type = 'DATETIME2'
                else:
                    sample = df[col].dropna().astype(str).str.len().max()
                    max_length = min(max(int(sample), 255), 4000) if pd.notnull(sample) else 255
                    col_type = f'NVARCHAR({max_length})'
                columns.append(f"[{col.replace(']', ']]')}] {col_type}")

        # Connect to database
        conn = None
        cursor = None
        try:
            if db_type == 'mysql':
                conn = mysql.connector.connect(**db_config)
                cursor = conn.cursor()
                quote_char = '`'
            elif db_type == 'postgres':
                conn = psycopg2.connect(**db_config)
                cursor = conn.cursor()
                quote_char = '"'
            elif db_type == 'sqlserver':
                conn_str = ';'.join([f"{k}={v}" for k, v in db_config.items()])
                conn = pyodbc.connect(conn_str)
                cursor = conn.cursor()
                quote_char = '['

            # Drop table if it exists to avoid conflicts
            safe_table_name = table_name.replace(quote_char, quote_char * 2)
            if db_type == 'mysql':
                drop_table_sql = f"DROP TABLE IF EXISTS `{safe_table_name}`"
            elif db_type == 'postgres':
                drop_table_sql = f'DROP TABLE IF EXISTS "{safe_table_name}"'
            elif db_type == 'sqlserver':
                drop_table_sql = f"IF OBJECT_ID('[{safe_table_name}]', 'U') IS NOT NULL DROP TABLE [{safe_table_name}]"
            logger.debug(f"Executing drop table SQL: {drop_table_sql}")
            cursor.execute(drop_table_sql)
            
            # Create table
            create_table_sql = f"CREATE TABLE {quote_char}{safe_table_name}{quote_char if db_type != 'sqlserver' else ']'} ({', '.join(columns)})"
            logger.debug(f"Executing create table SQL: {create_table_sql}")
            cursor.execute(create_table_sql)

            # Insert data
            placeholders = ', '.join(['?' if db_type == 'sqlserver' else '%s'] * len(df.columns))
            quoted_columns = [f"{quote_char}{col.replace(quote_char, quote_char * 2)}{quote_char if db_type != 'sqlserver' else ']'}" for col in df.columns]
            insert_sql = f"INSERT INTO {quote_char}{safe_table_name}{quote_char if db_type != 'sqlserver' else ']'} ({', '.join(quoted_columns)}) VALUES ({placeholders})"
            logger.debug(f"Insert SQL: {insert_sql}")

            for row in df.itertuples(index=False):
                cleaned_row = [None if pd.isna(val) else val for val in row]
                cursor.execute(insert_sql, tuple(cleaned_row))

            conn.commit()
        except (mysql.connector.Error, psycopg2.Error, pyodbc.Error) as db_error:
            logger.error(f"Database error: {str(db_error)}")
            raise Exception(f"Failed to connect to or process database: {str(db_error)}")
        except Exception as e:
            logger.error(f"Unexpected error during table creation: {str(e)}")
            raise Exception(f"Failed to process database operation: {str(e)}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
            
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return f"Successfully uploaded data to table {quote_char}{table_name}{quote_char if db_type != 'sqlserver' else ']'} in database {quote_char}{db_name}{quote_char if db_type != 'sqlserver' else ']'}."
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Error in create_table_from_file: {str(e)}")
        raise Exception(f"Failed to process file: {str(e)}")

@file_upload_bp.route('/upload_file', methods=['POST'])
def upload_file():
    """Handle file upload and create a table in the specified database."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file selected'}), 400

        db_name = request.form.get('db_name')
        table_name = request.form.get('table_name')
        db_type = request.form.get('db_type')

        if not db_name or not table_name or not db_type:
            return jsonify({'error': 'Missing required fields: db_name, table_name, or db_type'}), 400

        if db_type not in ['mysql', 'postgres', 'sqlserver']:
            return jsonify({'error': 'Invalid database type'}), 400

        message = create_table_from_file(file, db_name, table_name, db_type)
        return jsonify({'message': message})
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@file_upload_bp.route('/datapreview')
def datapreview():
    """Render the data preview page."""
    return render_template('datapreview.html')