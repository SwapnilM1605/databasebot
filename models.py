from database import db
from datetime import datetime
from flask_login import UserMixin

class Enterprise(db.Model):
    __bind_key__ = 'admin'
    __tablename__ = 'enterprise'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    mysql_host = db.Column(db.String(128))
    mysql_port = db.Column(db.Integer, default=3306)
    mysql_admin_user = db.Column(db.String(128))
    mysql_admin_password = db.Column(db.String(128))
    postgres_host = db.Column(db.String(128))
    postgres_port = db.Column(db.Integer, default=5432)
    postgres_admin_user = db.Column(db.String(128))
    postgres_admin_password = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    sqlserver_host = db.Column(db.String(128))
    sqlserver_port = db.Column(db.Integer, default=1433)
    sqlserver_instance = db.Column(db.String(128))  # New column for instance name
    sqlserver_admin_user = db.Column(db.String(128))
    sqlserver_admin_password = db.Column(db.String(128))
    
    # Relationships
    users = db.relationship('User', backref='user_enterprise', lazy=True)
    admin_users = db.relationship('AdminUser', backref='admin_enterprise', lazy=True)
    shared_databases = db.relationship('SharedDatabase', backref='database_enterprise', lazy=True)
    audit_logs = db.relationship('AdminLog', backref='log_enterprise', lazy=True)
    chat_threads = db.relationship('ChatThread', backref='thread_enterprise', lazy=True)

class User(db.Model, UserMixin):
    __bind_key__ = 'admin'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    mysql_password = db.Column(db.String(120))
    postgres_password = db.Column(db.String(120))
    sqlserver_password = db.Column(db.String(120))
    access = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    enterprise_id = db.Column(db.Integer, db.ForeignKey('enterprise.id'), nullable=False)
    
    __table_args__ = (
        db.UniqueConstraint('username', 'enterprise_id', name='unique_username_per_enterprise'),
    )

class AdminLog(db.Model):
    __bind_key__ = 'admin'
    __tablename__ = 'admin_log'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    action = db.Column(db.String(255), nullable=False)
    action_description = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    enterprise_id = db.Column(db.Integer, db.ForeignKey('enterprise.id'), nullable=False)
    
    enterprise = db.relationship('Enterprise', viewonly=True)

class AdminUser(db.Model):
    __bind_key__ = 'admin'
    __tablename__ = 'admin_user'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    enterprise_id = db.Column(db.Integer, db.ForeignKey('enterprise.id'), nullable=False)
    
    __table_args__ = (
        db.UniqueConstraint('username', 'enterprise_id', name='unique_admin_username_per_enterprise'),
    )

class ChatThread(db.Model):
    __bind_key__ = 'admin'
    __tablename__ = 'chat_thread'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(255))
    database = db.Column(db.String(255))
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime)
    enterprise_id = db.Column(db.Integer, db.ForeignKey('enterprise.id'), nullable=False)
    messages = db.relationship('ChatMessage', backref='message_thread', lazy=True)

    user = db.relationship('User', backref='user_threads')
    enterprise = db.relationship('Enterprise', viewonly=True)

class ChatMessage(db.Model):
    __bind_key__ = 'admin'
    __tablename__ = 'chat_message'
    id = db.Column(db.Integer, primary_key=True)
    thread_id = db.Column(db.Integer, db.ForeignKey('chat_thread.id'), nullable=False)
    sender = db.Column(db.Enum('user', 'bot'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime)
    database = db.Column(db.String(255))
    chart_data = db.Column(db.JSON)
    enterprise_id = db.Column(db.Integer, db.ForeignKey('enterprise.id'), nullable=False)

    thread = db.relationship('ChatThread', viewonly=True)
    enterprise = db.relationship('Enterprise', viewonly=True)

class SharedDatabase(db.Model):
    __bind_key__ = 'admin'
    __tablename__ = 'shared_database'
    
    id = db.Column(db.Integer, primary_key=True)
    database_name = db.Column(db.String(128), nullable=False)
    database_type = db.Column(db.Enum('mysql', 'postgres', 'sqlserver'), nullable=False, default='mysql')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    enterprise_id = db.Column(db.Integer, db.ForeignKey('enterprise.id'), nullable=False)
    
    __table_args__ = (
        db.UniqueConstraint('database_name', 'enterprise_id', name='uq_database_enterprise'),
    )
    
    def __repr__(self):
        return f'<SharedDatabase {self.database_name} ({self.database_type})>'

class ConfigSetting(db.Model):
    __bind_key__ = 'admin'
    __tablename__ = 'config_setting'
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
class Plan(db.Model):
    __bind_key__ = 'admin'
    __tablename__ = 'plan'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), nullable=False, unique=True)
    plan_name = db.Column(db.String(100), nullable=False)
    user_count = db.Column(db.Integer, nullable=False)
    price = db.Column(db.String(50), nullable=False)
    plan_type = db.Column(db.String(50), nullable=False)
    payment_status = db.Column(db.String(50), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class DataDictionary(db.Model):
    __bind_key__ = 'admin'
    __tablename__ = 'data_dictionary'
    
    id = db.Column(db.Integer, primary_key=True)
    table_name = db.Column(db.String(255), nullable=False)
    database_name = db.Column(db.String(255), nullable=False)
    database_type = db.Column(db.Enum('mysql', 'postgres', 'sqlserver'), nullable=False)
    dictionary = db.Column(db.JSON)  # Stores the data dictionary as JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    enterprise_id = db.Column(db.Integer, db.ForeignKey('enterprise.id'), nullable=False)
    
    __table_args__ = (
        db.UniqueConstraint('table_name', 'database_name', 'database_type', 'enterprise_id', 
                           name='uq_dictionary_table_database'),
    )
    
    enterprise = db.relationship('Enterprise', backref='dictionaries')