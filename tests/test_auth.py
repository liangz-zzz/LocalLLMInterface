from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient


def _stub_logger_module() -> types.ModuleType:
    class _Logger:
        def remove(self, *args, **kwargs):
            return None

        def add(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def debug(self, *args, **kwargs):
            return None

        def exception(self, *args, **kwargs):
            return None

    module = types.ModuleType("loguru")
    module.logger = _Logger()
    return module


def _stub_config_module() -> types.ModuleType:
    module = types.ModuleType("app.config")
    module.settings = SimpleNamespace(
        log_level="INFO",
        models_dir="/tmp/local-llm-test-models",
        gpu_memory_utilization=0.9,
        cors_origins=["*"],
        api_key=None,
        host="127.0.0.1",
        port=15530,
    )
    return module


def _stub_models_manager_module() -> types.ModuleType:
    class _ModelManager:
        def get_available_models(self):
            return {}

        async def cleanup(self):
            return None

    module = types.ModuleType("models.manager")
    module.model_manager = _ModelManager()
    return module


def _stub_router_module(*, include_health: bool = False) -> types.ModuleType:
    module = types.ModuleType("stub_router")
    router = APIRouter(prefix="/v1" if include_health else "")

    if include_health:
        @router.get("/health")
        async def _health():
            return {"status": "healthy"}

    module.router = router
    return module


@pytest.fixture
def client(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))

    for module_name in (
        "app.main",
        "app.config",
        "models.manager",
        "app.api.models",
        "app.api.chat",
        "app.api.embeddings",
        "app.api.rerank",
        "app.api.vision",
        "app.api.multimodal",
        "loguru",
    ):
        sys.modules.pop(module_name, None)

    monkeypatch.setitem(sys.modules, "loguru", _stub_logger_module())
    monkeypatch.setitem(sys.modules, "app.config", _stub_config_module())
    monkeypatch.setitem(sys.modules, "models.manager", _stub_models_manager_module())
    monkeypatch.setitem(sys.modules, "app.api.models", _stub_router_module(include_health=True))
    monkeypatch.setitem(sys.modules, "app.api.chat", _stub_router_module())
    monkeypatch.setitem(sys.modules, "app.api.embeddings", _stub_router_module())
    monkeypatch.setitem(sys.modules, "app.api.rerank", _stub_router_module())
    monkeypatch.setitem(sys.modules, "app.api.vision", _stub_router_module())
    monkeypatch.setitem(sys.modules, "app.api.multimodal", _stub_router_module())

    app_main = importlib.import_module("app.main")
    monkeypatch.setattr(app_main.settings, "api_key", "test-token")

    with TestClient(app_main.app) as test_client:
        yield test_client


def test_v1_health_requires_bearer_token(client):
    response = client.get("/v1/health")
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_v1_health_accepts_matching_bearer_token(client):
    response = client.get(
        "/v1/health",
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_non_v1_routes_stay_public(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["health_url"] == "/v1/health"
