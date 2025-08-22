from flask import Blueprint, request, session, jsonify
from models import db, ChatThread, ChatMessage, User
from datetime import datetime
import json

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/api/threads', methods=['GET'])
def get_threads():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user = User.query.filter_by(username=session['username']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    threads = ChatThread.query.filter_by(user_id=user.id).order_by(ChatThread.updated_at.desc()).all()
    thread_list = [
        {
            'id': t.id,
            'title': t.title,
            'created_at': t.created_at,
            'updated_at': t.updated_at
        }
        for t in threads
    ]
    return jsonify({'threads': thread_list})

@chat_bp.route('/api/threads', methods=['POST'])
def create_thread():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user = User.query.filter_by(username=session['username']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    data = request.get_json()
    title = data.get('title', '').strip()
    database = data.get('database', '').strip()
    db_type = data.get('type', '').strip()
    
    if not title:
        return jsonify({'error': 'Chat title is required'}), 400
    if not database:
        return jsonify({'error': 'Database name is required'}), 400
    if db_type not in ['mysql', 'postgres', 'sqlserver']:  # Added sqlserver
        return jsonify({'error': 'Invalid database type'}), 400
    
    # Extract database name from title (format: databaseName_userTitle)
    title_parts = title.split('_')
    if len(title_parts) < 2:
        return jsonify({'error': 'Invalid chat title format - must be "databaseName_userTitle"'}), 400
    
    db_name_from_title = title_parts[0]
    
    # Verify database matches title
    if db_name_from_title != database:
        return jsonify({'error': 'Title database name must match provided database'}), 400

    # Parse user.access as JSON
    try:
        access_list = json.loads(user.access) if user.access else []
    except json.JSONDecodeError as e:
        return jsonify({'error': f'Invalid access format in user record: {str(e)}'}), 500

    # Check if database is in user's access list
    if not any(db.get('name') == database and db.get('type') == db_type for db in access_list):
        # Check shared databases
        from main import load_shared_databases
        shared_databases = load_shared_databases()
        if not any(db.get('name') == database and db.get('type') == db_type for db in shared_databases):
            return jsonify({'error': f'No access to database "{database}"'}), 403

    now = datetime.utcnow()
    thread = ChatThread(
        user_id=user.id, 
        title=title, 
        created_at=now, 
        updated_at=now,
        database=database,
        enterprise_id=user.enterprise_id
    )
    db.session.add(thread)
    try:
        db.session.commit()
        return jsonify({
            'id': thread.id, 
            'title': thread.title, 
            'created_at': thread.created_at.isoformat(), 
            'updated_at': thread.updated_at.isoformat()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to create thread: {str(e)}'}), 500

@chat_bp.route('/api/threads/<int:thread_id>/messages', methods=['GET'])
def get_thread_messages(thread_id):
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user = User.query.filter_by(username=session['username']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    thread = ChatThread.query.filter_by(id=thread_id, user_id=user.id).first()
    if not thread:
        return jsonify({'error': 'Thread not found'}), 404

    messages = ChatMessage.query.filter_by(thread_id=thread_id).order_by(ChatMessage.timestamp.asc()).all()
    message_list = [
        {
            'id': m.id,
            'sender': m.sender,
            'content': m.content,
            'timestamp': m.timestamp.isoformat() if m.timestamp else None,
            'database': m.database,
            'chart_data': m.chart_data
        }
        for m in messages
    ]
    return jsonify({'messages': message_list})

@chat_bp.route('/api/threads/<int:thread_id>/messages', methods=['POST'])
def add_message(thread_id):
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user = User.query.filter_by(username=session['username']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    thread = ChatThread.query.filter_by(id=thread_id, user_id=user.id).first()
    if not thread:
        return jsonify({'error': 'Thread not found'}), 404

    data = request.get_json()
    sender = data.get('sender')
    content = data.get('content')
    database = data.get('database')
    chart_data = data.get('chart_data')

    if not sender or not content:
        return jsonify({'error': 'Sender and content are required'}), 400

    now = datetime.utcnow()
    message = ChatMessage(
        thread_id=thread_id, 
        sender=sender, 
        content=content, 
        timestamp=now,
        database=database,
        chart_data=chart_data,
        enterprise_id=user.enterprise_id  # Add enterprise_id here
    )
    
    thread.updated_at = now
    db.session.add(message)
    try:
        db.session.commit()
        return jsonify({
            'success': True, 
            'message_id': message.id,
            'timestamp': now.isoformat()
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to save message: {str(e)}'}), 500

@chat_bp.route('/api/threads/<int:thread_id>/context', methods=['GET'])
def get_thread_context(thread_id):
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user = User.query.filter_by(username=session['username']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    thread = ChatThread.query.filter_by(id=thread_id, user_id=user.id).first()
    if not thread:
        return jsonify({'error': 'Thread not found'}), 404

    database = request.args.get('database')
    if not database:
        return jsonify({'error': 'Database parameter is required'}), 400

    thread_db = thread.title.split('_')[0]
    if database != thread_db:
        return jsonify({'error': f'Context database does not match thread database'}), 400

    messages = ChatMessage.query.filter_by(
        thread_id=thread_id,
        database=database
    ).order_by(ChatMessage.timestamp.desc()).limit(5).all()

    message_list = [
        {
            'id': m.id,
            'sender': m.sender,
            'content': m.content,
            'timestamp': m.timestamp.isoformat() if m.timestamp else None,
            'database': m.database
        }
        for m in reversed(messages)
    ]
    return jsonify({'messages': message_list})

@chat_bp.route('/api/threads/<int:thread_id>', methods=['DELETE'])
def delete_thread(thread_id):
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user = User.query.filter_by(username=session['username']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    thread = ChatThread.query.filter_by(id=thread_id, user_id=user.id).first()
    if not thread:
        return jsonify({'error': 'Thread not found'}), 404

    try:
        ChatMessage.query.filter_by(thread_id=thread_id).delete()
        db.session.delete(thread)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Thread deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to delete thread: {str(e)}'}), 500
    
@chat_bp.route('/api/threads/<int:thread_id>/rename', methods=['POST'])
def rename_thread(thread_id):
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    user = User.query.filter_by(username=session['username']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404

    data = request.get_json()
    new_title = data.get('new_title')
    
    if not new_title or not new_title.strip():
        return jsonify({'error': 'Thread title cannot be empty'}), 400

    thread = ChatThread.query.filter_by(id=thread_id, user_id=user.id).first()
    if not thread:
        return jsonify({'error': 'Thread not found'}), 404

    parts = new_title.split('_')
    if len(parts) < 2:
        return jsonify({'error': 'Invalid title format'}), 400
        
    db_name = parts[0]
    access_list = user.access.split(',') if user.access else []
    if db_name not in access_list:
        return jsonify({'error': 'No access to specified database'}), 403

    thread.title = new_title.strip()
    try:
        db.session.commit()
        return jsonify({'success': True, 'new_title': thread.title})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500