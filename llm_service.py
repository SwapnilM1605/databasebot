from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.agent_toolkits import create_sql_agent
from config import Config
import os
import uuid
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import re
import ast
from datetime import datetime
from models import DataDictionary  # Import DataDictionary model

# PDF Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Forecasting imports
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class ForecastingService:
    def __init__(self):
        self.available_models = {
            'arima': self.arima_forecast,
            'prophet': self.prophet_forecast,
            'random_forest': self.random_forest_forecast
        }
    
    def prepare_time_series_data(self, df: pd.DataFrame, date_col: str, value_col: str) -> pd.Series:
        """Prepare time series data for forecasting"""
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            ts_data = df.set_index(date_col)[value_col]
            return ts_data
        except Exception as e:
            raise ValueError(f"Error preparing time series data: {str(e)}")
    
    def arima_forecast(self, ts_data: pd.Series, periods: int = 5) -> Dict[str, Any]:
        """ARIMA forecasting model"""
        try:
            model = ARIMA(ts_data, order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=periods)
            return {
                'model': 'ARIMA',
                'forecast': forecast.tolist(),
                'last_date': ts_data.index[-1],
                'metrics': {
                    'aic': model_fit.aic,
                    'bic': model_fit.bic
                }
            }
        except Exception as e:
            raise ValueError(f"ARIMA forecasting error: {str(e)}")
    
    def prophet_forecast(self, df: pd.DataFrame, date_col: str, value_col: str, periods: int = 5) -> Dict[str, Any]:
        """Facebook Prophet forecasting model"""
        try:
            prophet_df = df[[date_col, value_col]].rename(columns={date_col: 'ds', value_col: 'y'})
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            return {
                'model': 'Prophet',
                'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records'),
                'last_date': df[date_col].max(),
                'metrics': {
                    'trend_changes': len(model.changepoints)
                }
            }
        except Exception as e:
            raise ValueError(f"Prophet forecasting error: {str(e)}")
    
    def random_forest_forecast(self, df: pd.DataFrame, date_col: str, value_col: str, periods: int = 5) -> Dict[str, Any]:
        """Random Forest forecasting model"""
        try:
            # Feature engineering
            df['date'] = pd.to_datetime(df[date_col])
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            
            X = df[['year', 'month', 'day', 'day_of_week']]
            y = df[value_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train, y_train)
            
            # Generate future dates
            last_date = df['date'].max()
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq='D')[1:]
            
            future_df = pd.DataFrame({
                'date': future_dates,
                'year': future_dates.year,
                'month': future_dates.month,
                'day': future_dates.day,
                'day_of_week': future_dates.dayofweek
            })
            
            predictions = model.predict(future_df[['year', 'month', 'day', 'day_of_week']])
            
            return {
                'model': 'Random Forest',
                'forecast': [{'date': d, 'prediction': p} for d, p in zip(future_dates, predictions)],
                'last_date': last_date,
                'metrics': {
                    'mae': mean_absolute_error(y_test, model.predict(X_test)),
                    'feature_importance': dict(zip(X.columns, model.feature_importances_))
                }
            }
        except Exception as e:
            raise ValueError(f"Random Forest forecasting error: {str(e)}")
    
    def detect_best_model(self, df: pd.DataFrame, date_col: str, value_col: str) -> str:
        """Simple heuristic to suggest the best model based on data characteristics"""
        try:
            if len(df) < 30:
                return 'random_forest'  # Better for smaller datasets
            elif len(df) > 365:
                return 'prophet'  # Better for longer time series with seasonality
            else:
                return 'arima'  # Good general-purpose model
        except:
            return 'arima'  # Default fallback
    
    def forecast(self, data: Dict[str, Any], date_col: str, value_col: str, periods: int = 5, model: str = 'auto') -> Dict[str, Any]:
        """Main forecasting method"""
        try:
            df = pd.DataFrame(data)
            
            if model == 'auto':
                model = self.detect_best_model(df, date_col, value_col)
            
            if model not in self.available_models:
                raise ValueError(f"Unsupported model: {model}")
            
            if model == 'arima':
                ts_data = self.prepare_time_series_data(df, date_col, value_col)
                return self.available_models[model](ts_data, periods)
            else:
                return self.available_models[model](df, date_col, value_col, periods)
                
        except Exception as e:
            raise ValueError(f"Forecasting failed: {str(e)}")

class LLMService:
    def __init__(self):
        self.config = Config()
        self.llm = AzureChatOpenAI(
            model=self.config.AZURE_OPENAI_GPT_4_TURBO_MODEL,
            openai_api_key=self.config.AZURE_OPENAI_API_KEY,
            azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
            api_version=self.config.AZURE_API_VERSION_GPT_4
        )
        self.forecasting_service = ForecastingService()

    def load_chat_history_from_db(self, thread_id: int, db_session) -> List[Dict[str, Any]]:
        """Load chat history from database for a specific thread."""
        if not db_session:
            return []
            
        try:
            # First verify the database connection
            db_session.execute("SELECT 1")
            
            query = """
            SELECT sender, content, database, timestamp 
            FROM chat_message
            WHERE thread_id = %s
            ORDER BY timestamp ASC
            """
            db_session.execute(query, (thread_id,))
            rows = db_session.fetchall()
            
            return [
                {
                    "sender": row["sender"],
                    "content": row["content"],
                    "database": row["database"],
                    "timestamp": row["timestamp"]
                }
                for row in rows
            ]
        except Exception as e:
            return []

    def get_data_dictionary(self, db: SQLDatabase, database_name: str, enterprise_id: int) -> str:
        """Fetch data dictionary for tables in the database if available."""
        try:
            # Get table names from the database
            table_info = db.get_table_info()
            table_names = re.findall(r'\b\w+\b(?=\s+\()', table_info)  # Extract table names from schema
            
            # Query DataDictionary for matching tables
            dictionaries = []
            for table_name in table_names:
                dictionary = DataDictionary.query.filter_by(
                    database_name=database_name,
                    table_name=table_name,
                    enterprise_id=enterprise_id
                ).first()
                if dictionary and dictionary.dictionary:
                    dictionaries.append({
                        "table_name": table_name,
                        "dictionary": dictionary.dictionary
                    })
            
            if not dictionaries:
                return "No data dictionary available."
            
            # Format the dictionaries as a string
            formatted_dict = "\n".join([
                f"Table: {d['table_name']}\nDictionary: {json.dumps(d['dictionary'], indent=2)}"
                for d in dictionaries
            ])
            return formatted_dict
        except Exception as e:
            return f"Error fetching data dictionary: {str(e)}"

    def get_sql_chain(self, db, enterprise_id: int, database_name: str):
        template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema and data dictionary below, write a SQL query that would answer the user's question.

        <SCHEMA>{schema}</SCHEMA>
        <DATA_DICTIONARY>{data_dictionary}</DATA_DICTIONARY>

        Previous Context (if relevant):
        {chat_history}

        Current Question: {question}

        Instructions:
        1. Use the data dictionary to understand column meanings and ensure accurate column selection in queries.
        2. If the question asks for forecasting, predictions, or future trends:
        - Return historical data with date and value columns
        - Ensure the query returns data in chronological order
        - Include enough historical data for meaningful forecasting (at least 12 data points if available)
        3. For regular questions:
        - Consider the full conversation context
        - If the current question refers to previous results, account for that context
        4. Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
        5. For visualization requests, ensure the query returns data in a format suitable for charts

        Your turn:

        Question: {question}
        SQL Query:
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Get schema and dictionary synchronously first
        schema = db.get_table_info()
        data_dictionary = self.get_data_dictionary(db, database_name, enterprise_id)
        
        return (
            {
                "question": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
                "schema": lambda x: schema,
                "data_dictionary": lambda x: data_dictionary
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def should_visualize(self, query: str, sql_response: str) -> Tuple[bool, str]:
        """
        Determine if the query should be visualized and what type of chart to use.
        Returns a tuple of (should_visualize, chart_type)
        """
        template = """
        Analyze if the following query and its response should be visualized as a chart.
        If yes, determine the most appropriate chart type.

        Query: {query}
        SQL Response: {sql_response}

        Instructions:
        1. Determine if this data would be better understood through visualization
        2. If yes, specify the chart type from these options: 
        line, bar, horizontal_bar, stacked_bar, pie, doughnut, area, scatter, bubble
        3. Consider:
        - If the query mentions "chart", "graph", "visualize", "plot" → use appropriate chart type
        - If the query asks for comparisons or rankings → bar or stacked_bar chart
        - If the query asks for trends over time → line or area chart
        - If the query asks for proportions or percentages → pie or doughnut chart
        - If the query asks for relationships between variables → scatter or bubble chart
        - If the data has 3 dimensions (x, y, size) → bubble chart
        - For horizontal comparisons → horizontal_bar
        4. Return only the chart type or 'none' if no visualization is needed

        Chart type:
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        chart_chain = (
            RunnablePassthrough()
            | prompt
            | self.llm
            | StrOutputParser()
        )

        chart_type = chart_chain.invoke({
            "query": query,
            "sql_response": sql_response
        }).strip().lower()

        # If the query explicitly mentions chart or visualization, force chart creation
        query_lower = query.lower()
        if any(word in query_lower for word in ['chart', 'graph', 'visualize', 'plot', 'bar', 'line', 'pie', 'stacked', 'doughnut', 'area', 'bubble', 'scatter']):
            if 'bar' in query_lower or 'stacked' in query_lower:
                if 'horizontal' in query_lower:
                    return True, 'horizontal_bar'
                return True, 'stacked_bar' if 'stacked' in query_lower else 'bar'
            elif 'line' in query_lower:
                return True, 'line'
            elif 'area' in query_lower:
                return True, 'area'
            elif 'pie' in query_lower or 'doughnut' in query_lower:
                return True, 'doughnut' if 'doughnut' in query_lower else 'pie'
            elif 'scatter' in query_lower:
                return True, 'scatter'
            elif 'bubble' in query_lower:
                return True, 'bubble'
            else:
                return True, 'bar'  # Default to bar chart if visualization is requested but type not specified

        return chart_type != 'none', chart_type
    
    def transform_to_chart_data(self, sql_response: str, chart_type: str) -> Dict[str, Any]:
        """
        Transform SQL response into Chart.js compatible JSON format with enhanced visuals and interactivity.
        Skips rows with NaN values to ensure valid JSON output.
        """
        try:
            # Convert SQL response to pandas DataFrame
            from io import StringIO
            import pandas as pd
            
            # Parse the JSON response
            df = pd.read_json(StringIO(sql_response))
            
            # Drop rows with any NaN values
            df = df.dropna()
            
            # Validate DataFrame: Ensure it has at least one row and two columns
            if df.empty:
                return None
            if len(df.columns) < 2:
                return None
            
            # Get column names for labels
            x_label = df.columns[0]
            y_label = df.columns[1]
            
            # Color palette for charts
            colors = [
                'rgba(54, 162, 235, 0.7)', 'rgba(255, 99, 132, 0.7)', 
                'rgba(75, 192, 192, 0.7)', 'rgba(255, 159, 64, 0.7)',
                'rgba(153, 102, 255, 0.7)', 'rgba(255, 205, 86, 0.7)'
            ]
            
            # Base chart configuration
            chart_data = {
                "type": chart_type if chart_type != 'horizontal_bar' else 'bar',  # horizontal_bar is a bar with indexAxis
                "data": {
                    "labels": df[x_label].tolist(),
                    "datasets": []
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "plugins": {
                        "legend": {
                            "position": "top",
                            "labels": {
                                "font": {
                                    "size": 14
                                }
                            }
                        },
                        "title": {
                            "display": True,
                            "text": f"{y_label} by {x_label}",
                            "font": {
                                "size": 18
                            }
                        },
                        "tooltip": {
                            "enabled": True,
                            "backgroundColor": "rgba(0,0,0,0.8)",
                            "titleFont": {
                                "size": 16
                            },
                            "bodyFont": {
                                "size": 14
                            }
                        }
                    },
                    "animation": {
                        "duration": 1000,
                        "easing": "easeOutQuart"
                    },
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "title": {
                                "display": True,
                                "text": y_label,
                                "font": {
                                    "size": 14
                                }
                            }
                        },
                        "x": {
                            "title": {
                                "display": True,
                                "text": x_label,
                                "font": {
                                    "size": 14
                                }
                            }
                        }
                    }
                }
            }

            # For horizontal bar chart, swap the axes
            if chart_type == 'horizontal_bar':
                chart_data["options"]["indexAxis"] = "y"
                chart_data["options"]["scales"]["x"]["title"]["text"] = y_label
                chart_data["options"]["scales"]["y"]["title"]["text"] = x_label

            # Chart-specific configurations
            if chart_type in ['bar', 'horizontal_bar', 'stacked_bar']:
                dataset = {
                    "label": y_label,
                    "data": df[y_label].tolist(),
                    "backgroundColor": colors[:len(df)],
                    "borderColor": [c.replace('0.7', '1') for c in colors[:len(df)]],
                    "borderWidth": 1
                }
                if chart_type == 'stacked_bar' and len(df.columns) > 2:
                    chart_data["data"]["datasets"] = [
                        {
                            "label": df.columns[i],
                            "data": df[df.columns[i]].tolist(),
                            "backgroundColor": colors[i % len(colors)],
                            "borderColor": colors[i % len(colors)].replace('0.7', '1'),
                            "borderWidth": 1
                        } for i in range(1, len(df.columns))
                    ]
                    chart_data["options"]["scales"]["y"]["stacked"] = True
                    chart_data["options"]["scales"]["x"]["stacked"] = True
                else:
                    chart_data["data"]["datasets"] = [dataset]
                    
            elif chart_type == 'line':
                chart_data["data"]["datasets"] = [{
                    "label": y_label,
                    "data": df[y_label].tolist(),
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 2,
                    "fill": False,
                    "tension": 0.4,
                    "pointRadius": 5,
                    "pointHoverRadius": 8
                }]
                chart_data["options"]["elements"] = {
                    "line": {
                        "borderWidth": 3
                    },
                    "point": {
                        "radius": 4,
                        "hoverRadius": 6
                    }
                }
                
            elif chart_type == 'area':
                chart_data["data"]["datasets"] = [{
                    "label": y_label,
                    "data": df[y_label].tolist(),
                    "backgroundColor": "rgba(54, 162, 235, 0.5)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 2,
                    "fill": True,
                    "tension": 0.4,
                    "pointRadius": 5,
                    "pointHoverRadius": 8
                }]
                chart_data["options"]["elements"] = {
                    "line": {
                        "borderWidth": 3
                    },
                    "point": {
                        "radius": 4,
                        "hoverRadius": 6
                    }
                }
                    
            elif chart_type in ['pie', 'doughnut']:
                chart_data["data"]["datasets"] = [{
                    "label": y_label,
                    "data": df[y_label].tolist(),
                    "backgroundColor": colors[:len(df)],
                    "borderColor": ['#ffffff'] * len(df),
                    "borderWidth": 2
                }]
                chart_data["options"]["plugins"]["legend"]["position"] = "right"
                if chart_type == 'doughnut':
                    chart_data["options"]["cutout"] = "60%"
                    
            elif chart_type == 'scatter':
                chart_data["data"]["datasets"] = [{
                    "label": y_label,
                    "data": [
                        {"x": row[x_label], "y": row[y_label]}
                        for _, row in df.iterrows()
                    ],
                    "backgroundColor": "rgba(255, 99, 132, 0.7)",
                    "borderColor": "rgba(255, 99, 132, 1)",
                    "borderWidth": 1,
                    "pointRadius": 6,
                    "pointHoverRadius": 8
                }]
                
            elif chart_type == 'bubble':
                # For bubble chart, we need x, y, and size (r) values
                if len(df.columns) >= 3:
                    size_label = df.columns[2]
                    chart_data["data"]["datasets"] = [{
                        "label": y_label,
                        "data": [
                            {
                                "x": row[x_label],
                                "y": row[y_label],
                                "r": row[size_label] / df[size_label].max() * 20  # Normalize size
                            }
                            for _, row in df.iterrows()
                        ],
                        "backgroundColor": "rgba(255, 99, 132, 0.7)",
                        "borderColor": "rgba(255, 99, 132, 1)",
                        "borderWidth": 1
                    }]
                    chart_data["options"]["scales"]["x"]["title"]["text"] = x_label
                    chart_data["options"]["scales"]["y"]["title"]["text"] = y_label
                else:
                    # Fall back to scatter if we don't have size data
                    chart_data["type"] = "scatter"
                    chart_data["data"]["datasets"] = [{
                        "label": y_label,
                        "data": [
                            {"x": row[x_label], "y": row[y_label]}
                            for _, row in df.iterrows()
                        ],
                        "backgroundColor": "rgba(255, 99, 132, 0.7)",
                        "borderColor": "rgba(255, 99, 132, 1)",
                        "borderWidth": 1,
                        "pointRadius": 6,
                        "pointHoverRadius": 8
                    }]

            return chart_data
        except Exception as e:
            return None

    def format_response(self, response_text: str) -> str:
        """
        Format the response text with proper HTML formatting for display.
        Enhanced to better handle tables, lists, and other structured data.
        """
        # First check if this is a markdown table
        if self._is_markdown_table(response_text):
            return self._format_enhanced_table(response_text)
        
        # Check for bullet points or numbered lists
        if any(line.strip().startswith(('- ', '* ', '• ', '1. ', '2. ')) 
                for line in response_text.split('\n')):
            return self._format_lists(response_text)

        # Check for code blocks
        if "```" in response_text:
            return self._format_code_blocks(response_text)

        # Default formatting for plain text
        return f"<div style='line-height: 1.6;'>{response_text}</div>"

    def _format_enhanced_table(self, text: str) -> str:
        """Convert markdown table to a well-formatted HTML table with enhanced styling"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if len(lines) < 3:  # Not a proper table
            return f"<div style='line-height: 1.6;'>{text}</div>"
        
        # Extract headers
        headers = [h.strip() for h in lines[0].split("|") if h.strip()]
        
        # Build HTML table with enhanced styling
        table_html = """
        <div class="enhanced-table-container">
            <table class="enhanced-table">
                <thead>
                    <tr>"""
        
        for header in headers:
            table_html += f"<th>{header}</th>"
        
        table_html += """
                    </tr>
                </thead>
                <tbody>"""
        
        # Add rows (skip the separator line)
        for line in lines[2:]:
            if "|" in line:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) == len(headers):
                    table_html += "<tr>"
                    for i, cell in enumerate(cells):
                        # Format numbers with thousands separators
                        formatted_cell = self._format_numeric_cell(cell)
                        
                        # Apply different styling for header column vs data columns
                        if i == 0:
                            table_html += f'<td class="header-column">{formatted_cell}</td>'
                        else:
                            table_html += f'<td class="data-column">{formatted_cell}</td>'
                    table_html += "</tr>"
        
        table_html += """
                </tbody>
            </table>
        </div>"""
        
        return table_html

    def _format_numeric_cell(self, cell: str) -> str:
        """Format numeric cells with thousands separators if applicable"""
        try:
            # Try to format as integer
            num = int(cell.replace(',', ''))
            return f"{num:,}"
        except ValueError:
            try:
                # Try to format as float
                num = float(cell.replace(',', ''))
                return f"{num:,.2f}" if num % 1 else f"{int(num):,}"
            except ValueError:
                return cell

    def _is_markdown_table(self, text: str) -> bool:
        """Check if text contains a markdown table"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 2:
            return False
        return all("|" in line for line in lines[:3]) and "---" in lines[1]

    def _format_markdown_table(self, text: str) -> str:
        """Convert markdown table to HTML with enhanced styling"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Extract headers
        headers = [h.strip() for h in lines[0].split("|") if h.strip()]
        
        # Build HTML table with proper structure and styling
        table_html = """
        <div class='response-table-container'>
            <table class='response-table'>
                <thead>
                    <tr>"""
        
        for header in headers:
            table_html += f"<th>{header}</th>"
        
        table_html += """
                    </tr>
                </thead>
                <tbody>"""
        
        # Add rows (skip the separator line)
        for line in lines[2:]:
            if "|" in line:
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if len(cells) == len(headers):
                    table_html += "<tr>"
                    for i, cell in enumerate(cells):
                        # Right-align numeric cells
                        is_numeric = False
                        try:
                            float(cell.replace(',', ''))
                            is_numeric = True
                        except ValueError:
                            pass
                        
                        if is_numeric:
                            table_html += f"<td class='numeric'>{cell}</td>"
                        else:
                            table_html += f"<td>{cell}</td>"
                    table_html += "</tr>"
        
        table_html += """
                </tbody>
            </table>
        </div>"""
        
        return table_html

    def _format_lists(self, text: str) -> str:
        """Format bullet points and numbered lists with proper HTML"""
        lines = text.split('\n')
        in_list = False
        html = "<div style='line-height: 1.6;'>"
        
        for line in lines:
            stripped = line.strip()
            
            # Check for list items
            if stripped.startswith(('- ', '* ', '• ', '1. ', '2. ', '3. ')):
                if not in_list:
                    html += "<ul>" if any(stripped.startswith(x) for x in ('- ', '* ', '• ')) else "<ol>"
                    in_list = True
                # Remove the bullet/number
                content = stripped[2:].strip()
                html += f"<li>{content}</li>"
            else:
                if in_list:
                    html += "</ul>" if any(stripped.startswith(x) for x in ('- ', '* ', '• ')) else "</ol>"
                    in_list = False
                if stripped:
                    html += f"<p>{line}</p>"
        
        if in_list:
            html += "</ul>" if any(stripped.startswith(x) for x in ('- ', '* ', '• ')) else "</ol>"
        
        html += "</div>"
        return html

    def _format_code_blocks(self, text: str) -> str:
        """Format code blocks with proper HTML"""
        parts = text.split("```")
        formatted = ""
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Code block
                # Remove language specifier if present
                code = part.split('\n', 1)[-1] if '\n' in part else part
                formatted += f"<pre><code>{code}</code></pre>"
            else:
                formatted += part.replace("\n", "<br>")
        return formatted

    def get_response(self, user_query: str, db: SQLDatabase = None, vectorstore: Chroma = None, chat_history: list = [], enterprise_id: int = None, database_name: str = None):
        if vectorstore is not None:
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

            template = """
            You are a data analyst. Convert the given SQL response into a user-friendly natural language response.

            Previous Context (if relevant):
            {chat_history}

            Current Question: {question}
            SQL Response: {sql_response}

            Instructions for tabular data:
            - Always format as a proper markdown table with clear headers
            - Use exactly two columns for category/value displays
            - Keep category names in the first column
            - Put numeric values in the second column
            - Sort data logically (alphabetically or by value as appropriate)
            - Include all relevant data without truncation

            Example format:
            | Category       | Count |
            |----------------|-------|
            | Category A     | 100   |
            | Category B     | 200   |

            Now provide the response:
            """
            prompt = ChatPromptTemplate.from_template(template)

            from langchain.chains import RetrievalQA

            # Get contextual question with chat history
            full_query = self.get_contextual_question(chat_history, user_query)

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=False
            )

            try:
                result = qa_chain.run(full_query)
                formatted_result = self.format_response(result.strip())
                return formatted_result, None
            except Exception as e:
                return f"<div class='error-message'>Error during retrieval-based QA: {str(e)}</div>", None

        # Format chat history for context
        formatted_history = ""
        if chat_history:
            formatted_history = "\n".join([
                f"{'User' if msg['sender'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in chat_history
            ])

        # Get SQL query
        sql_chain = self.get_sql_chain(db, enterprise_id, database_name)
        sql_query = sql_chain.invoke({
            "question": user_query,
            "chat_history": formatted_history
        })
        
        try:
            # Execute SQL query
            sql_response = db.run(sql_query)
            
            # Check if we should visualize this response
            should_visualize, chart_type = self.should_visualize(user_query, str(sql_response))
            
            # Format the response into natural language
            template = """
            You are a data analyst. Convert the given SQL response into a user-friendly natural language response.

            Previous Context (if relevant):
            {chat_history}

            Current Question: {question}
            SQL Response: {sql_response}

            Instructions:
            - Format the response in a natural, conversational way
            - If the answer is a table, format it using markdown table format
            - If the answer is descriptive or summary-based, format it into a short paragraph
            - Never include the SQL query or mention SQL in the response
            - Keep the response concise and easy to understand
            - If this is a follow-up question, ensure the response relates to previous context
            - Use markdown formatting for better readability (bold, italics, lists, tables)
            - For numerical data, include units and context
            - For comparisons, highlight key differences
            - For trends, describe the pattern clearly

            Now give the final response:
            """

            prompt = ChatPromptTemplate.from_template(template)

            response_formatter = (
                RunnablePassthrough()
                | prompt
                | self.llm
                | StrOutputParser()
            )

            final_nl_response = response_formatter.invoke({
                "question": user_query,
                "sql_response": str(sql_response),
                "chat_history": formatted_history
            })
            
            # Format the response with HTML
            formatted_response = self.format_response(final_nl_response.strip())
            
            # If visualization is needed, transform the data
            chart_data = None
            if should_visualize:
                try:
                    # Convert the SQL response to a DataFrame
                    import pandas as pd
                    from io import StringIO
                    
                    if isinstance(sql_response, str):
                        try:
                            # Convert string representation of list of tuples to actual list of tuples
                            import ast
                            data = ast.literal_eval(sql_response)
                            # Get column names from the SQL query
                            import re
                            column_names = re.findall(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE)[0]
                            column_names = [col.strip().split()[-1] for col in column_names.split(',')]
                            df = pd.DataFrame(data, columns=column_names)
                        except Exception as e:
                            df = pd.read_json(StringIO(sql_response))
                    else:
                        df = pd.DataFrame(sql_response)
                    
                    # Convert to JSON for chart transformation
                    json_data = df.to_json(orient='records')
                    chart_data = self.transform_to_chart_data(json_data, chart_type)
                except Exception as e:
                    chart_data = None
            
            return formatted_response, chart_data

        except Exception as e:
            return f"<div class='error-message'>Error executing SQL query: {str(e)}</div>", None

    def get_contextual_question(self, chat_history: list, current_question: str) -> str:
        """Create a contextual question based on chat history."""
        if not chat_history:
            return current_question

        context = "\n".join([
            f"{'User' if msg['sender'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in chat_history[-3:]  # Use last 3 messages for context
        ])

        return f"""Previous conversation:
{context}

Current question: {current_question}"""
    
    def process_pdf(self, pdf_file):
        """Process a PDF file and create a vector store."""
        try:
            # Load PDF
            loader = PyPDFLoader(pdf_file)
            pages = loader.load()

            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(pages)

            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=AzureOpenAIEmbeddings(
                    openai_api_key=self.config.AZURE_OPENAI_API_KEY,
                    azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
                    api_version=self.config.AZURE_API_VERSION_GPT_4
                )
            )

            return vectorstore
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return None