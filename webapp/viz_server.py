"""The Flask application: routes that wire the browser to the model service."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from attnviz import Config

from .image_codec import ImageCodec
from .model_service import ModelService
from .session_store import Session, SessionStore


class VizServer:
    """Builds and serves the attention-visualization web app.

    Three JSON endpoints back the page: ``/api/generate`` (prompt -> image +
    tokens), ``/api/cross`` (token -> heatmap overlay) and ``/api/self``
    (image point -> heatmap overlay). Generations are cached in a
    :class:`SessionStore` so token/region lookups never rerun the model.
    """

    def __init__(self, config: Config):
        self._app = Flask(__name__)
        self._service = ModelService(config)
        self._store = SessionStore()
        self._codec = ImageCodec()
        self._register_routes()

    @property
    def app(self) -> Flask:
        return self._app

    def run(self, host: str = "127.0.0.1", port: int = 5000, debug: bool = False) -> None:
        self._app.run(host=host, port=port, debug=debug, threaded=True)

    def _register_routes(self) -> None:
        self._app.add_url_rule("/", "index", self._index)
        self._app.add_url_rule("/api/generate", "generate", self._generate, methods=["POST"])
        self._app.add_url_rule("/api/load_model", "load_model", self._load_model, methods=["POST"])
        self._app.add_url_rule("/api/cross", "cross", self._cross, methods=["POST"])
        self._app.add_url_rule("/api/self", "self_attn", self._self, methods=["POST"])
        self._app.add_url_rule("/api/image2text", "image2text", self._image2text, methods=["POST"])
        self._app.register_error_handler(KeyError, self._on_missing_session)
        self._app.register_error_handler(Exception, self._on_error)

    def _on_missing_session(self, _exc):
        return jsonify({"error": "Session expired — please generate again."}), 410

    def _on_error(self, exc):
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    def _index(self):
        return render_template(
            "index.html",
            device=self._service.device_label,
            models=self._service.available_models(),
            loaded_model=self._service.loaded_model_id,
        )

    def _generate(self):
        data = request.get_json(force=True)
        session = self._service.generate(
            model_id=self._model_id(data),
            prompt=str(data.get("prompt", "")).strip() or "a photograph",
            steps=int(data.get("steps", 30)),
            seed=int(data.get("seed", 0)),
            size=int(data.get("size", 512)),
            guidance=float(data.get("guidance", 7.5)),
        )
        session_id = self._store.add(session)
        return jsonify(self._generation_payload(session_id, session))

    def _load_model(self):
        data = request.get_json(force=True)
        model_id = self._model_id(data)
        loaded = self._service.ensure_model(model_id)
        return jsonify({"model_id": model_id, "loaded": loaded,
                        "device": self._service.device_label})

    def _model_id(self, data) -> str:
        model_id = str(data.get("model_id") or self._service.default_model_id())
        if not self._service.is_allowed(model_id):
            raise ValueError(f"Unknown model: {model_id}")
        return model_id

    def _cross(self):
        data = request.get_json(force=True)
        session = self._lookup(data)
        alpha = float(data.get("alpha", 0.6))
        source = data.get("source") or "score"
        overlay = session.cross.overlay(
            int(data["token_index"]), alpha=alpha, layer=self._layer(data), source=source
        )
        return jsonify({"overlay": self._codec.array_to_data_url(overlay)})

    def _self(self):
        data = request.get_json(force=True)
        session = self._lookup(data)
        alpha = float(data.get("alpha", 0.6))
        overlay = session.self_attn.overlay(
            int(data["x"]), int(data["y"]), alpha=alpha, layer=self._layer(data)
        )
        return jsonify({"overlay": self._codec.array_to_data_url(overlay)})

    def _image2text(self):
        data = request.get_json(force=True)
        session = self._lookup(data)
        weights = session.image2text.token_weights(
            int(data["x"]), int(data["y"]), session.result.image.size[0],
            layer=self._layer(data),
            include_special=bool(data.get("include_special", False)),
        )
        return jsonify({"tokens": [
            {"index": w.index, "label": w.label, "weight": w.weight, "special": w.special}
            for w in weights
        ]})

    def _lookup(self, data) -> Session:
        return self._store.get(str(data["session_id"]))

    def _layer(self, data):
        """Read the optional layer name; empty/missing means 'average'."""
        layer = data.get("layer")
        return layer or None

    def _generation_payload(self, session_id: str, session: Session) -> dict:
        cross = session.cross
        tokens = [{"index": i, "label": cross.display_label(i),
                   "special": cross.is_special(i)}
                  for i in cross.displayable_token_indices()]
        return {
            "session_id": session_id,
            "image": self._codec.pil_to_data_url(session.result.image),
            "width": session.result.image.size[0],
            "prompt": session.result.prompt,
            "tokens": tokens,
            "layers": cross.available_layers(),
            "self_layers": session.self_attn.available_layers(),
            "cross_res": cross.grid_res,
            "self_res": session.self_attn.resolution,
            "device": self._service.device_label,
            "model": self._service.loaded_model_id,
        }
