import os
from dotenv import load_dotenv
import psycopg2

# Load environment variables from .env file
load_dotenv()

_connection = None

def get_connection():
    """Establish and return a connection to the PostgreSQL database."""
    global _connection
    if _connection is None or _connection.closed != 0:
        _connection = psycopg2.connect(
            database=os.getenv("DB_NAME", "healthcare_agent"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
    return _connection

# Pre-initialized connection instance for tools.py
connection = get_connection()
