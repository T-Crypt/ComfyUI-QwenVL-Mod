"""Offline reliability tests for the QwenChat panel backend.

No GPU, no network: the pack's __init__.py is exec'd against a stub
PromptServer so route registration is checked for real (the c963f53 refactor
once silently deleted POST /qwenvl/chat, GET /qwenvl/chat/models and
GET /qwenvl/chat/assets), and every llama-swap / ComfyUI HTTP call is a
unittest.mock. Behavior under test: the /free payload (it used to be an empty
body, a silent no-op), the warm-card guard, the occupied-card refusal, and
what a llama-swap failure turns into for the panel.
"""

import asyncio
import importlib.util
import io
import json
import sys
import unittest
import urllib.error
import urllib.request
import unittest.mock
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[1]))

PACK_ROOT = Path(__file__).parents[1]

# The three routes a refactor once swept out of __init__.py. Any of these
# disappearing must fail this suite.
CHAT_ROUTES = {
    ("POST", "/qwenvl/chat"): "_chat",
    ("GET", "/qwenvl/chat/models"): "_chat_models",
    ("GET", "/qwenvl/chat/tools"): "_chat_tools",
    ("GET", "/qwenvl/chat/assets"): "_chat_assets",
}

SIBLING_ROUTES = {
    ("POST", "/qwenvl/chat/unload"): "_chat_unload",
    ("GET", "/qwenvl/models/search"): "_models_search",
    ("POST", "/qwenvl/models/download"): "_models_download",
}

# Sibling node modules import torch / transformers at import time; the route
# handlers under test do not need them. Stubbing keeps the exec to ~1s and
# offline (load_modules_from_directory catches the resulting stub errors).
HEAVY_IMPORTS = ("numpy", "psutil", "torch", "huggingface_hub", "transformers")


def _load_pack_module():
    """Exec the pack's __init__.py with `server` and `folder_paths` stubbed,
    return (module, registered routes as {(method, path): handler})."""
    added = []
    saved = {}
    for name in list(sys.modules):
        saved[name] = sys.modules[name]

    def install(name, module):
        if name not in sys.modules:
            sys.modules[name] = module
            added.append(name)

    for name in HEAVY_IMPORTS:
        install(name, MagicMock())

    class _Routes:
        def __init__(self):
            self.registered = {}

        def _add(self, method, path):
            def deco(handler):
                self.registered[(method, path)] = handler
                return handler
            return deco

        def get(self, path):
            return self._add("GET", path)

        def post(self, path):
            return self._add("POST", path)

    routes = _Routes()
    server_stub = ModuleType("server")
    server_stub.PromptServer = type(
        "PromptServer", (), {"instance": __import__("types").SimpleNamespace(routes=routes)})
    install("server", server_stub)

    folder_paths_stub = ModuleType("folder_paths")
    folder_paths_stub.get_output_directory = lambda: "/tmp"
    install("folder_paths", folder_paths_stub)

    spec = importlib.util.spec_from_file_location("qwenvl_pack_under_test", PACK_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name in added:
            sys.modules.pop(name, None)
        stale = {name: mod for name, mod in sys.modules.items() if name not in saved}
        for name in stale:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
    return module, routes.registered


def _json_response(payload):
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    return response


def _fake_request(body):
    async def read_json():
        return body
    request = MagicMock()
    request.json = read_json
    return request


def _url_and_method(call):
    # Some call sites pass a bare URL string, others a urllib Request.
    if isinstance(call, str):
        return call, "GET"
    return call.full_url, call.get_method()


def _http_error(code, reason, body=b""):
    return urllib.error.HTTPError(
        "http://under-test", code, reason, None, io.BytesIO(body))


class RouteRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.module, self.registered = _load_pack_module()

    def test_chat_routes_are_registered(self):
        for route, handler_name in CHAT_ROUTES.items():
            with self.subTest(route=route):
                handler = self.registered.get(route)
                self.assertIsNotNone(handler, f"missing route {route[0]} {route[1]}")
                self.assertEqual(handler.__name__, handler_name)

    def test_sibling_chat_routes_are_registered(self):
        for route, handler_name in SIBLING_ROUTES.items():
            with self.subTest(route=route):
                self.assertIn(route, self.registered)

    def test_chat_reports_upstream_error_to_the_panel(self):
        # A llama-swap 502 becomes a ValueError in _api_chat_completions; the
        # route must surface that message (HTTP 400 + body), not swallow it.
        def failing_chat(*args, **kwargs):
            raise ValueError("llama-swap API error HTTP 502: upstream command exited prematurely")

        with patch.object(self.module, "CHAT_RUNTIME", MagicMock(chat=failing_chat)):
            response = asyncio.run(self.module._chat(_fake_request(
                {"backend": "api", "model": "m", "messages": [{"role": "user", "content": "hi"}]})))
        self.assertEqual(response.status, 400)
        self.assertIn("HTTP 502", response.text)

    def test_chat_maps_unexpected_failure_to_500_with_generic_message(self):
        def broken_chat(*args, **kwargs):
            raise KeyError("protocol bug")

        with patch.object(self.module, "CHAT_RUNTIME", MagicMock(chat=broken_chat)):
            response = asyncio.run(self.module._chat(_fake_request(
                {"backend": "api", "model": "m", "messages": [{"role": "user", "content": "hi"}]})))
        self.assertEqual(response.status, 500)
        self.assertIn("Chat generation failed", response.text)


class ComfyFreeTests(unittest.TestCase):
    """Ping-pong guard: POST /free must carry its flags, poll until the card
    frees, refuse when occupied with llama-swap empty, and never block a warm
    chat model."""

    def test_free_posts_flags_and_polls_until_freed(self):
        # Regression: the original version POSTed an empty body, which
        # ComfyUI's /free route 500s on an empty body, so this was a silent
        # no-op.
        calls = []

        def urlopen(request, timeout=None):
            calls.append(request)
            url, method = _url_and_method(request)
            if method == "POST" and url.endswith("/free"):
                return _json_response({})
            return _json_response({"devices": [{"vram_free": 24 * 2**30, "vram_total": 24 * 2**30}]})

        from chat_service import _comfy_free
        with patch("urllib.request.urlopen", side_effect=urlopen):
            _comfy_free("NInfer-HauHauCS-27B")
        # Sequence: the warm-model check (GET /running), POST /free, then one
        # /system_stats poll (the first already shows the card freed).
        self.assertEqual(len(calls), 3)
        self.assertTrue(_url_and_method(calls[0])[0].endswith("/running"))
        free_url, free_method = _url_and_method(calls[1])
        self.assertTrue(free_url.endswith("/free"))
        self.assertEqual(free_method, "POST")
        payload = json.loads(calls[1].data.decode("utf-8"))
        self.assertIs(payload["unload_models"], True)
        self.assertIs(payload["free_memory"], True)
        self.assertTrue(_url_and_method(calls[2])[0].endswith("/system_stats"))

    def test_warm_model_skips_the_free_guard(self):
        from chat_service import _comfy_free
        with patch("urllib.request.urlopen") as urlopen, \
                patch("chat_service._llamaswap_running", return_value={"NInfer-HauHauCS-27B"}):
            _comfy_free("NInfer-HauHauCS-27B")
        urlopen.assert_not_called()

    def test_occupied_card_with_empty_llamaswap_refuses(self):
        # The llama-server would die on the load ("upstream command exited
        # prematurely"); the guard must stop with an actionable error instead.
        from chat_service import _comfy_free
        urlopen = unittest.mock.MagicMock(side_effect=[
            _json_response({}),  # GET /running -> unexpected shape, fail-open
            _json_response({}),  # POST /free accepted
            _json_response({"devices": [{"vram_free": 1 * 2**30, "vram_total": 24 * 2**30}]}),
            _json_response({"running": []}),  # llama-swap holds nothing it could swap out
        ])
        with patch("urllib.request.urlopen", urlopen), \
                patch("chat_service.FREE_WAIT_S", 0):
            with self.assertRaises(ValueError) as caught:
                _comfy_free("NInfer-HauHauCS-27B")
        self.assertIn("Restart ComfyUI", str(caught.exception))

    def test_unreadable_stats_fail_open(self):
        from chat_service import _comfy_free
        urlopen = unittest.mock.MagicMock(side_effect=[
            _json_response({}),  # GET /running -> unexpected shape, fail-open
            _json_response({}),  # POST /free accepted
            urllib.error.URLError("system_stats hiccup"),
        ])
        with patch("urllib.request.urlopen", urlopen):
            _comfy_free("NInfer-HauHauCS-27B")  # must not raise


class ApiChatCompletionsTests(unittest.TestCase):
    """What a llama-swap failure (502/503 during a cold model load, dead
    service, timeout) turns into: a ValueError whose message reaches the
    panel."""

    def _attempt(self, error):
        from chat_service import _api_chat_completions
        # Single attempt: these cases pin the error mapping, not the retry.
        with patch("urllib.request.urlopen", side_effect=error), \
                patch("chat_service.API_RETRY_ATTEMPTS", 0):
            _api_chat_completions("system", "user", [], "NInfer-HauHauCS-27B", {"max_tokens": 8})

    def test_http_502_maps_to_actionable_value_error(self):
        with self.assertRaises(ValueError) as caught:
            self._attempt(_http_error(502, "Bad Gateway", b"upstream command exited prematurely"))
        self.assertIn("HTTP 502", str(caught.exception))
        self.assertIn("upstream command exited prematurely", str(caught.exception))

    def test_http_503_maps_to_actionable_value_error(self):
        with self.assertRaises(ValueError) as caught:
            self._attempt(_http_error(503, "Service Unavailable", b"loading model"))
        self.assertIn("HTTP 503", str(caught.exception))
        self.assertIn("loading model", str(caught.exception))

    def test_dead_llamaswap_maps_to_reachability_hint(self):
        with self.assertRaises(ValueError) as caught:
            self._attempt(urllib.error.URLError("Connection refused"))
        self.assertIn("not reachable", str(caught.exception))
        self.assertIn("Is llama-swap running?", str(caught.exception))

    def test_timeout_maps_to_retry_hint(self):
        with self.assertRaises(ValueError) as caught:
            self._attempt(TimeoutError())
        self.assertIn("timed out", str(caught.exception))


class ColdLoadRetryTests(unittest.TestCase):
    """Finding 2: a chat that races a cold NInfer load must retry the fast
    llama-swap failure instead of surfacing it to the panel."""

    def _call(self):
        from chat_service import _api_chat_completions
        return _api_chat_completions("system", "user", [], "NInfer-HauHauCS-27B", {"max_tokens": 8})

    def test_502_retries_then_returns_the_reply(self):
        calls = []

        def urlopen(request, timeout=None):
            calls.append(request)
            if len(calls) < 3:
                raise _http_error(502, "Bad Gateway", b"upstream command exited prematurely")
            return _json_response({"choices": [{"message": {"content": '{"message":"hi","actions":[]}'}}]})

        with patch("urllib.request.urlopen", side_effect=urlopen), \
                patch("chat_service.API_RETRY_ATTEMPTS", 2), \
                patch("chat_service.API_RETRY_BACKOFF_S", 0.0):
            text, finish = self._call()
        self.assertIn('"message":"hi"', text)
        self.assertEqual(len(calls), 3)

    def test_502_exhausts_retries_and_names_the_failure(self):
        with patch("urllib.request.urlopen",
                   side_effect=_http_error(502, "Bad Gateway", b"upstream command exited prematurely")), \
                patch("chat_service.API_RETRY_ATTEMPTS", 2), \
                patch("chat_service.API_RETRY_BACKOFF_S", 0.0):
            with self.assertRaises(ValueError) as caught:
                self._call()
        self.assertIn("HTTP 502", str(caught.exception))

    def test_500_without_the_respawn_hint_does_not_retry(self):
        calls = []

        def urlopen(request, timeout=None):
            calls.append(request)
            raise _http_error(500, "Internal Server Error", b"real bug")

        with patch("urllib.request.urlopen", side_effect=urlopen), \
                patch("chat_service.API_RETRY_ATTEMPTS", 2), \
                patch("chat_service.API_RETRY_BACKOFF_S", 0.0):
            with self.assertRaises(ValueError):
                self._call()
        self.assertEqual(len(calls), 1)

    def test_500_with_the_respawn_hint_retries(self):
        calls = []

        def urlopen(request, timeout=None):
            calls.append(request)
            if len(calls) == 1:
                raise _http_error(500, "Internal Server Error", b"upstream command exited prematurely")
            return _json_response({"choices": [{"message": {"content": '{"message":"ok","actions":[]}'}}]})

        with patch("urllib.request.urlopen", side_effect=urlopen), \
                patch("chat_service.API_RETRY_ATTEMPTS", 2), \
                patch("chat_service.API_RETRY_BACKOFF_S", 0.0):
            text, _ = self._call()
        self.assertIn('"message":"ok"', text)
        self.assertEqual(len(calls), 2)

    def test_connection_refused_retries(self):
        calls = []

        def urlopen(request, timeout=None):
            calls.append(request)
            if len(calls) == 1:
                raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
            return _json_response({"choices": [{"message": {"content": '{"message":"ok","actions":[]}'}}]})

        with patch("urllib.request.urlopen", side_effect=urlopen), \
                patch("chat_service.API_RETRY_ATTEMPTS", 2), \
                patch("chat_service.API_RETRY_BACKOFF_S", 0.0):
            text, _ = self._call()
        self.assertIn('"message":"ok"', text)
        self.assertEqual(len(calls), 2)

    def test_timeout_is_not_retried(self):
        calls = []

        def urlopen(request, timeout=None):
            calls.append(request)
            raise TimeoutError()

        with patch("urllib.request.urlopen", side_effect=urlopen), \
                patch("chat_service.API_RETRY_ATTEMPTS", 2), \
                patch("chat_service.API_RETRY_BACKOFF_S", 0.0):
            with self.assertRaises(ValueError) as caught:
                self._call()
        self.assertIn("timed out", str(caught.exception))
        self.assertEqual(len(calls), 1)

    def test_retryable_status_helper(self):
        from chat_service import _api_retryable
        self.assertTrue(_api_retryable(502, ""))
        self.assertTrue(_api_retryable(503, "loading model"))
        self.assertFalse(_api_retryable(400, "bad request"))
        self.assertFalse(_api_retryable(500, "disk full"))
        self.assertTrue(_api_retryable(500, "Upstream command exited prematurely"))


class ApiModelListTests(unittest.TestCase):
    """The selector list comes from llama-swap, keeps the default first, and
    never blanks when llama-swap is down."""

    def setUp(self):
        from chat_service import _api_models_cache
        _api_models_cache.update(at=0.0, names=[])

    def test_default_is_first_and_embeddings_are_dropped(self):
        payload = {"data": [{"id": "ZZZ"}, {"id": "NInfer-HauHauCS-27B"},
                            {"id": "text-embedding-nomic-embed-text-v1.5"}]}
        from chat_service import _api_models
        with patch("urllib.request.urlopen", return_value=_json_response(payload)):
            names = _api_models()
        self.assertEqual(names[0], "NInfer-HauHauCS-27B")
        self.assertIn("ZZZ", names)
        self.assertNotIn("text-embedding-nomic-embed-text-v1.5", names)

    def test_falls_back_to_the_default_when_llama_swap_is_down(self):
        from chat_service import _api_models, API_DEFAULT_MODEL
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            names = _api_models()
        self.assertEqual(names, [API_DEFAULT_MODEL])

    def test_cache_avoids_a_second_socket_call(self):
        from chat_service import _api_models
        with patch("urllib.request.urlopen", return_value=_json_response({"data": [{"id": "ZZZ"}]})) as urlopen:
            _api_models()
            _api_models()
        self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
