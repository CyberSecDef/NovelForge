"""NovelForge – Flask backend for AI-powered novel generation."""
from dotenv import load_dotenv
load_dotenv()

from novelforge import create_app

app = create_app()

if __name__ == "__main__":
    import os
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    app.run(debug=True, host=host, port=port)
