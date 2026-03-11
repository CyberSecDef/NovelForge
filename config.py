"""
Configuration for NovelForge.

Settings are read from environment variables. Copy .env.example to .env
and fill in your values before running the application.
"""

import os


# LLM API endpoint (default: OpenAI chat completions)
LLM_API_URL = os.environ.get(
    "LLM_API_URL", "https://api.openai.com/v1/chat/completions"
)

# LLM API key – REQUIRED in production. Set the LLM_API_KEY environment variable.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")

# LLM model name to request
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")

# Flask secret key – override via SECRET_KEY environment variable in production
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")

# Directory where Flask-Session stores server-side session files
SESSION_FILE_DIR = os.environ.get("SESSION_FILE_DIR", "./flask_session")

# Directory where exported novel files are stored temporarily
EXPORT_DIR = os.environ.get("EXPORT_DIR", "./exports")
