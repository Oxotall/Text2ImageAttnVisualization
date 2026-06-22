"""Flask web UI for interactive attention visualization.

Components:
* ImageCodec     — encode PIL images / numpy overlays as base64 data URLs,
* SessionStore   — keep per-generation results and visualizers in memory,
* ModelService   — lazily load the model and run generations,
* VizServer      — the Flask application wiring routes to the service.
"""

from .image_codec import ImageCodec
from .session_store import SessionStore, Session
from .model_service import ModelService
from .viz_server import VizServer

__all__ = ["ImageCodec", "SessionStore", "Session", "ModelService", "VizServer"]
