"""Route blueprints for NovelForge."""

from flask import Flask


def register_blueprints(app: Flask) -> None:
    from novelforge.routes.outline import outline_bp
    from novelforge.routes.generation import generation_bp
    from novelforge.routes.export import export_bp
    from novelforge.routes.sessions import sessions_bp
    app.register_blueprint(outline_bp)
    app.register_blueprint(generation_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(sessions_bp)
