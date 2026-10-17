import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from chat_service import (
    ChatRuntime,
    MINIMAX_I2VA_BINDING,
    THINK_CLOSE,
    THINK_OPEN,
    build_prompt,
    enforce_image_enhancer_routing,
    enforce_image_reference_bindings,
    list_output_images,
    parse_model_response,
    select_workflow_intent,
    validate_actions,
    validate_graph,
    validate_images,
    validate_messages,
    _fix_minimax_preset_actions,
    _match_minimax_preset,
    _minimax_preset_mode,
    _minimax_result,
    _is_image_enhancer_node,
    _has_image_enhancer_target,
    ensure_i2va_binding,
    normalize_minimax_output,
)


class ChatProtocolTests(unittest.TestCase):
    def test_validates_and_limits_messages(self):
        messages = [{"role": "user", "content": f"message {index}"} for index in range(25)]
        result = validate_messages(messages)
        self.assertEqual(len(result), 20)
        self.assertEqual(result[0]["content"], "message 5")

    def test_rejects_invalid_message_role_and_content(self):
        for messages in ([], [{"role": "system", "content": "x"}], [{"role": "user", "content": ""}]):
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                validate_messages(messages)

    def test_truncates_overlong_message_content(self):
        from chat_service import MAX_MESSAGE_CHARS
        long_text = "y" * (MAX_MESSAGE_CHARS + 500)
        result = validate_messages([{"role": "user", "content": long_text}])
        self.assertEqual(result[0]["content"], "y" * MAX_MESSAGE_CHARS)

    def test_fit_history_to_budget_keeps_latest(self):
        from chat_service import PROMPT_BUDGET_CHARS, fit_history_to_budget
        old = [{"role": "user", "content": f"old-{index}-" + "z" * 50000} for index in range(10)]
        latest = {"role": "user", "content": "the actual request"}
        fitted = fit_history_to_budget(old + [latest])
        self.assertEqual(fitted[-1]["content"], "the actual request")
        self.assertLessEqual(sum(len(m["content"]) for m in fitted), PROMPT_BUDGET_CHARS)
        self.assertLess(len(fitted), 11)

    def test_api_reasoning_content_surfaces_as_thinking(self):
        import json as _json
        from unittest import mock
        import chat_service as cs

        payload = {"choices": [{"message": {
            "content": '{"message":"hi","actions":[],"choices":[]}',
            "reasoning_content": "because reasons",
        }}]}

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return _json.dumps(payload).encode()

        with mock.patch.object(cs.urllib.request, "urlopen", return_value=_Resp()):
            text, finish = cs._api_chat_completions("SYS", "PROMPT", [], "M", {"max_tokens": 64})
        thinking, rest = cs.extract_thinking(text)
        self.assertEqual(thinking, "because reasons")
        self.assertIn('"message"', rest)
        self.assertEqual(finish, "")

    def test_api_chat_completions_reports_truncated_finish_reason(self):
        import json as _json
        from unittest import mock
        import chat_service as cs

        payload = {"choices": [{"finish_reason": "length", "message": {"content": '{"message":"cut'}}]}

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return _json.dumps(payload).encode()

        with mock.patch.object(cs.urllib.request, "urlopen", return_value=_Resp()):
            text, finish = cs._api_chat_completions("SYS", "PROMPT", [], "M", {"max_tokens": 64})
        self.assertEqual(finish, "length")
        self.assertIn("cut", text)

    def test_chat_retry_escalates_budget_and_seed_on_truncation(self):
        from unittest import mock
        import chat_service as cs

        calls = []

        def fake_generate(self, backend, model_name, prompt, options, enable_thinking, images, video=None):
            calls.append({"prompt": prompt, "options": dict(options)})
            if len(calls) < 3:
                return '{"message":"fixing","actions":[{"type":"set_widget_value","node_id":12,"wid', "length"
            return '{"message":"done","actions":[],"choices":[]}', "stop"

        runtime = cs.ChatRuntime()
        with mock.patch.object(cs.ChatRuntime, "_generate", fake_generate):
            result = runtime.chat("api", None, [{"role": "user", "content": "fix all four KSamplers"}], {"nodes": []}, {"max_tokens": 1024, "seed": 7})
        self.assertTrue(result["parsed"])
        self.assertEqual(calls[0]["options"]["max_tokens"], 4096)
        self.assertGreaterEqual(calls[1]["options"]["max_tokens"], 8192)
        self.assertGreaterEqual(calls[2]["options"]["max_tokens"], 16384)
        self.assertNotEqual(calls[1]["options"]["seed"], 7)
        self.assertIn("cut off", calls[1]["prompt"])
        self.assertIn("cut off", calls[2]["prompt"])

    def test_accepts_structural_actions(self):
        actions = validate_actions([
            {"type": "add_node", "key": "k1", "node_type": "KSampler", "title": "Hi", "x": 10, "y": 20.5},
            {"type": "remove_node", "node_id": 3},
            {"type": "connect", "from": "k1", "out": "LATENT", "to": 2, "in": 0},
            {"type": "disconnect", "node_id": 2, "input": "model"},
            {"type": "layout"},
        ])
        self.assertEqual(len(actions), 5)
        self.assertEqual(actions[0]["key"], "k1")
        self.assertEqual(actions[0]["x"], 10)
        self.assertEqual(actions[2]["from"], "k1")

    def test_rejects_malformed_structural_actions(self):
        self.assertEqual(validate_actions([
            {"type": "add_node", "key": "", "node_type": "KSampler"},
            {"type": "add_node", "node_type": "KSampler"},
            {"type": "connect", "from": 1, "to": 2},
            {"type": "disconnect", "node_id": 2},
            {"type": "teleport"},
        ]), [])

    def test_validates_graph_links_and_pos(self):
        graph = {"nodes": [{
            "id": 1, "type": "KSampler", "title": "K", "mode": 0, "widgets": [],
            "inputs": [{"name": "model", "type": "MODEL", "link": 7}, {"name": "bad"}],
            "outputs": [{"name": "LATENT", "type": "LATENT", "links": [9, "x"]}],
            "pos": [100, 200],
        }, {
            "id": 2, "type": "X", "title": "X", "mode": 0, "widgets": [],
            "pos": "elsewhere",
        }]}
        result = validate_graph(graph)
        self.assertEqual(result["nodes"][0]["inputs"][0]["link"], 7)
        self.assertEqual(len(result["nodes"][0]["inputs"]), 2)
        self.assertEqual(result["nodes"][0]["inputs"][1], {"name": "bad", "type": "", "link": None})
        self.assertEqual(result["nodes"][0]["outputs"][0]["links"], [9])
        self.assertEqual(result["nodes"][0]["pos"], [100, 200])
        self.assertEqual(result["nodes"][1]["pos"], [0, 0])
        self.assertEqual(result["nodes"][1].get("outputs"), [])

    def test_validates_graph_snapshot(self):
        graph = {"nodes": [{"id": 1, "type": "KSampler", "title": "Sampler", "mode": 0, "widgets": [{"name": "steps", "value": 20}]}]}
        result = validate_graph(graph)
        self.assertEqual(result["nodes"][0]["id"], 1)
        self.assertEqual(result["nodes"][0]["widgets"][0]["value"], 20)

    def test_rejects_invalid_graph(self):
        for graph in (None, {}, {"nodes": "bad"}, {"nodes": [{}]}):
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                validate_graph(graph)

    def test_accepts_only_protocol_actions(self):
        actions = [
            {"type": "set_widget_value", "node_id": 1, "widget": "steps", "value": 25},
            {"type": "set_node_mode", "node_id": 2, "mode": "bypass"},
            {"type": "queue_workflow"},
            {"type": "delete_node", "node_id": 3},
            {"type": "set_node_mode", "node_id": 2, "mode": "delete"},
            {"type": "set_widget_value", "node_id": 1, "widget": "x", "value": {"unsafe": True}},
        ]
        self.assertEqual(len(validate_actions(actions)), 3)

    def test_parses_json_and_fenced_json(self):
        payload = {"message": "Done", "actions": [{"type": "queue_workflow"}]}
        for text in (json.dumps(payload), f"```json\n{json.dumps(payload)}\n```"):
            with self.subTest(text=text):
                result = parse_model_response(text)
                self.assertEqual(result["message"], payload["message"])
                self.assertEqual(result["actions"], payload["actions"])
                self.assertEqual(result["thinking"], "")

    def test_extracts_thinking_from_response(self):
        text = THINK_OPEN + "I should queue the workflow." + THINK_CLOSE + "\n" + json.dumps(
            {"message": "Done", "actions": [{"type": "queue_workflow"}]}
        )
        result = parse_model_response(text)
        self.assertEqual(result["thinking"], "I should queue the workflow.")
        self.assertEqual(result["message"], "Done")
        self.assertEqual(len(result["actions"]), 1)

    def test_plain_text_becomes_message_without_actions(self):
        self.assertEqual(parse_model_response("I cannot do that"), {"message": "I cannot do that", "actions": [], "choices": [], "thinking": "", "parsed": False})

    def test_prompt_contains_history_and_graph(self):
        prompt = build_prompt([{"role": "user", "content": "Set steps"}], {"nodes": [{"id": 1}]})
        self.assertIn("Set steps", prompt)
        self.assertIn('"id":1', prompt)
        self.assertIn("set_widget_value", prompt)
        self.assertIn("images", prompt)
        self.assertIn("Workflow prompt text must be English", prompt)
        self.assertIn("mirror the LATEST user message language", prompt)
        self.assertIn("MiniMax H3 video sampler", prompt)
        self.assertIn("include queue_workflow in the same response", prompt)

    def test_prompt_identifies_exact_promoted_passthrough_target(self):
        graph = {"nodes": [{"id": 105, "widgets": [
            {"name": "prompt", "value": ""},
            {"name": "preset_prompt", "value": "MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": False},
        ]}]}
        prompt = build_prompt([{"role": "user", "content": "Generate the video"}], graph)
        self.assertIn('Node 105 exposes both "prompt" and "passthrough"', prompt)
        self.assertIn('set node 105 widget "prompt"', prompt)
        self.assertIn('set node 105 widget "passthrough" to true', prompt)

    def test_prompt_uses_inner_enhancer_when_image_is_provided(self):
        graph = {"nodes": [{"id": 105, "title": "Image to Video (MiniMax H3)", "widgets": [
            {"name": "prompt", "value": ""},
            {"name": "preset_prompt", "value": "MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": True},
        ]}]}
        prompt = build_prompt([{"role": "user", "content": "Generate the video"}], graph, has_images=True)
        self.assertIn("Image pixels are provided", prompt)
        self.assertIn("Inspect the provided image pixels to understand how the requested action applies", prompt)
        self.assertIn('currently selects preset "MiniMax H3 NSFW (5s)"', prompt)
        self.assertIn("IGNORE any full prompt-writing guide for that preset", prompt)
        self.assertIn("the inner QwenVL node will use it to build the final prompt", prompt)
        self.assertIn('You MUST set node 105 widget "prompt" to a concise English action directive', prompt)
        self.assertIn('set node 105 widget "passthrough" to false', prompt)
        self.assertIn("inner QwenVL must analyze the image and create the final preset prompt", prompt)
        # The full MiniMax format guide must not leak into the chat prompt for an image enhancer.
        self.assertNotIn("integrated_multimodal_description:", prompt)
        self.assertNotIn("overall_soundscape:", prompt)

    def test_selects_previous_intent_after_execution_confirmation(self):
        descriptive = "Create a five-second video where she opens the dress"
        for confirmation in (
            "Generate the video", "Esegui il video", "Run it", "Genera", "Ok",
            "Sì", "Avvia", "Esegui il video con il prompt precedente", "Go ahead",
        ):
            messages = [
                {"role": "user", "content": descriptive},
                {"role": "assistant", "content": "Ready."},
                {"role": "user", "content": confirmation},
            ]
            with self.subTest(confirmation=confirmation):
                self.assertEqual(select_workflow_intent(messages), descriptive)

    def test_execution_only_detection_keeps_descriptive_requests(self):
        for content in (
            "genera un video in cui la ragazza balla",
            "crea un video di 5 secondi dall'immagine allegata",
            "la ragazza sposta la mano e si afferra il seno",
            "modifica il prompt per dire che lei sorride",
        ):
            with self.subTest(content=content):
                self.assertEqual(select_workflow_intent([{"role": "user", "content": content}]), content)

    def test_enforces_image_enhancer_prompt_and_disables_passthrough(self):
        graph = {"nodes": [{"id": 105, "title": "Image to Video (MiniMax H3)", "widgets": [
            {"name": "prompt", "value": "old prompt"},
            {"name": "preset_prompt", "value": "🎬 MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": True},
        ]}]}
        messages = [
            {"role": "user", "content": "La ragazza apre il vestito"},
            {"role": "assistant", "content": "Vuoi che lo esegua?"},
            {"role": "user", "content": "Esegui il video"},
        ]
        cases = [
            ([], messages[0]["content"]),
            ([{"type": "set_widget_value", "node_id": 105, "widget": "prompt", "value": "integrated_multimodal_description: [Shot 1] hallucinated scene"}], messages[0]["content"]),
            ([{"type": "set_widget_value", "node_id": 105, "widget": "prompt", "value": "Esegui il video"}], messages[0]["content"]),
        ]
        for prompt_action, expected in cases:
            with self.subTest(prompt_action=prompt_action):
                result = {
                    "message": "Done",
                    "actions": [
                        {"type": "set_widget_value", "node_id": 105, "widget": "preset_prompt", "value": "🎬 MiniMax H3 NSFW (5s)"},
                        *prompt_action,
                        {"type": "set_widget_value", "node_id": 105, "widget": "passthrough", "value": True},
                        {"type": "queue_workflow"},
                    ],
                }
                enforced = enforce_image_enhancer_routing(result, graph, messages, True)
                prompt_actions = [action for action in enforced["actions"] if action.get("widget") == "prompt"]
                self.assertEqual(len(prompt_actions), 1)
                self.assertEqual(prompt_actions[0]["value"], expected)
                self.assertFalse(next(action for action in enforced["actions"] if action.get("widget") == "passthrough")["value"])

    def test_keeps_model_directive_for_image_enhancer(self):
        graph = {"nodes": [{"id": 105, "title": "Image to Video (MiniMax H3)", "widgets": [
            {"name": "prompt", "value": "old prompt"},
            {"name": "preset_prompt", "value": "🎬 MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": True},
        ]}]}
        messages = [{"role": "user", "content": "la ragazza sposta la mano e si afferra il seno"}]
        directive = "The woman moves her hand and grabs her breast. Preserve the reference image exactly and change only this action."
        result = {
            "message": "Done",
            "actions": [
                {"type": "set_widget_value", "node_id": 105, "widget": "prompt", "value": directive},
                {"type": "set_widget_value", "node_id": 105, "widget": "passthrough", "value": True},
                {"type": "queue_workflow"},
            ],
        }
        enforced = enforce_image_enhancer_routing(result, graph, messages, True)
        prompt_action = next(action for action in enforced["actions"] if action.get("widget") == "prompt")
        self.assertEqual(prompt_action["value"], directive)
        self.assertIn(directive, enforced["message"])
        self.assertFalse(next(action for action in enforced["actions"] if action.get("widget") == "passthrough")["value"])

    def test_enforces_minimax_i2va_binding_for_image_passthrough(self):
        result = {
            "message": "Prompt generated.",
            "actions": [
                {"type": "set_widget_value", "node_id": 105, "widget": "prompt", "value": "integrated_multimodal_description: [Shot 1] action"},
                {"type": "set_widget_value", "node_id": 105, "widget": "passthrough", "value": True},
                {"type": "queue_workflow"},
            ],
            "choices": [],
            "thinking": "",
        }
        graph = {"nodes": [{"id": 105, "title": "Image to Video (MiniMax H3)", "widgets": [
            {"name": "prompt", "value": ""},
            {"name": "preset_prompt", "value": "🎬 MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": False},
        ]}]}
        enforced = enforce_image_reference_bindings(result, graph, True)
        self.assertTrue(enforced["actions"][0]["value"].startswith(MINIMAX_I2VA_BINDING))
        self.assertIn(MINIMAX_I2VA_BINDING, enforced["message"])

    def test_skips_minimax_binding_when_passthrough_set_false(self):
        """When the action set flips passthrough to false, do not prepend the I2VA binding
        even if the workflow snapshot still has passthrough=true."""
        result = {
            "message": "Prompt generated.",
            "actions": [
                {"type": "set_widget_value", "node_id": 105, "widget": "prompt", "value": "the woman touches her nipple"},
                {"type": "set_widget_value", "node_id": 105, "widget": "passthrough", "value": False},
                {"type": "queue_workflow"},
            ],
            "choices": [],
            "thinking": "",
        }
        graph = {"nodes": [{"id": 105, "title": "Image to Video (MiniMax H3)", "widgets": [
            {"name": "prompt", "value": ""},
            {"name": "preset_prompt", "value": "🎬 MiniMax H3 NSFW (5s)"},
            {"name": "passthrough", "value": True},
        ]}]}
        enforced = enforce_image_reference_bindings(result, graph, True)
        self.assertNotIn(MINIMAX_I2VA_BINDING, enforced["actions"][0]["value"])
        self.assertNotIn(MINIMAX_I2VA_BINDING, enforced["message"])

    def test_validates_images(self):
        import base64
        valid = base64.b64encode(b"fake-image-data").decode("ascii")
        self.assertEqual(len(validate_images([valid, "not-valid", 123])), 1)
        self.assertEqual(len(validate_images([valid, valid, valid, valid])), 3)

    def test_video_frames_use_higher_limit(self):
        import base64
        frame = base64.b64encode(b"frame").decode("ascii")
        self.assertEqual(len(validate_images([frame] * 5, 4)), 4)
        self.assertEqual(len(validate_images([frame] * 5)), 3)

    def test_prompt_mentions_video_frames(self):
        prompt = build_prompt([{"role": "user", "content": "make it faster"}], {"nodes": []}, has_video=True)
        self.assertIn("VIDEO INPUT", prompt)
        prompt_no_video = build_prompt([{"role": "user", "content": "hi"}], {"nodes": []})
        self.assertNotIn("VIDEO INPUT", prompt_no_video)

    def _jpeg_b64(self):
        import base64
        import io
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (128, 64, 32)).save(buffer, "JPEG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def test_hf_chat_forwards_video_frames(self):
        class FakeQuantization:
            Q8 = types.SimpleNamespace(value="q8")

        class FakeBase:
            def load_model(self, *args):
                pass

            def generate(self, *args, **kwargs):
                self.kwargs = kwargs
                self.frame_count = args[4]
                return '{"message":"ok","actions":[]}'

        previous = sys.modules.get("AILab_QwenVL")
        sys.modules["AILab_QwenVL"] = types.SimpleNamespace(
            HF_ALL_MODELS={"test-model": {}},
            Quantization=FakeQuantization,
            QwenVLBase=FakeBase,
        )
        try:
            runtime = ChatRuntime()
            frames = [self._jpeg_b64() for _ in range(4)]
            result = runtime.chat("hf", "test-model", [{"role": "user", "content": "make it faster"}], {"nodes": []}, {}, video=frames)
            self.assertEqual(result["message"], "ok")
            instance = runtime._instances["hf"]
            self.assertEqual(len(instance.kwargs["video"]), 4)
            self.assertEqual(instance.frame_count, 4)
        finally:
            if previous is None:
                sys.modules.pop("AILab_QwenVL", None)
            else:
                sys.modules["AILab_QwenVL"] = previous

    def test_gguf_chat_reencodes_video_frames_as_images(self):
        class FakeBase:
            def _load_model(self, *args):
                pass

            def _invoke(self, system, prompt, images_b64, *args, **kwargs):
                self.images_b64 = images_b64
                return '{"message":"ok","actions":[]}'

        previous = sys.modules.get("AILab_QwenVL_GGUF")
        sys.modules["AILab_QwenVL_GGUF"] = types.SimpleNamespace(
            GGUF_VL_CATALOG={"models": {"test-gguf": {}}},
            QwenVLGGUFBase=FakeBase,
        )
        try:
            runtime = ChatRuntime()
            frames = [self._jpeg_b64() for _ in range(4)]
            result = runtime.chat("gguf", "test-gguf", [{"role": "user", "content": "make it faster"}], {"nodes": []}, {}, video=frames)
            self.assertEqual(result["message"], "ok")
            instance = runtime._instances["gguf"]
            self.assertEqual(len(instance.images_b64), 4)
            # Frames must round-trip as valid base64 strings (not raw bytes)
            import base64
            for item in instance.images_b64:
                self.assertIsInstance(item, str)
                base64.b64decode(item, validate=True)
        finally:
            if previous is None:
                sys.modules.pop("AILab_QwenVL_GGUF", None)
            else:
                sys.modules["AILab_QwenVL_GGUF"] = previous

    def test_lists_nested_output_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "root.png").write_bytes(b"png")
            (root / "PMP" / "2026-09-16").mkdir(parents=True)
            (root / "PMP" / "2026-09-16" / "nested.webp").write_bytes(b"webp")
            (root / "ignored.mp4").write_bytes(b"video")
            (root / "ignored.txt").write_text("text")
            assets = list_output_images(root)
        self.assertEqual(set(assets), {"root.png [output]", "PMP/2026-09-16/nested.webp [output]", "ignored.mp4 [output]"})

    def test_hf_runtime_reuses_and_unloads_model(self):
        class FakeQuantization:
            Q8 = types.SimpleNamespace(value="q8")

        class FakeBase:
            def __init__(self):
                self.loaded = 0
                self.cleared = False

            def load_model(self, *args):
                self.loaded += 1

            def generate(self, *args, **kwargs):
                return '{"message":"ok","actions":[]}'

            def clear(self):
                self.cleared = True

        previous = sys.modules.get("AILab_QwenVL")
        sys.modules["AILab_QwenVL"] = types.SimpleNamespace(
            HF_ALL_MODELS={"test-model": {}},
            Quantization=FakeQuantization,
            QwenVLBase=FakeBase,
        )
        try:
            runtime = ChatRuntime()
            result = runtime.chat("hf", "test-model", [{"role": "user", "content": "hello"}], {"nodes": []}, {})
            self.assertEqual(result["message"], "ok")
            instance = runtime._instances["hf"]
            self.assertEqual(instance.loaded, 1)
            self.assertEqual(runtime.unload("hf"), ["hf"])
            self.assertTrue(instance.cleared)
        finally:
            if previous is None:
                sys.modules.pop("AILab_QwenVL", None)
            else:
                sys.modules["AILab_QwenVL"] = previous

    def test_minimax_preset_mode_detection(self):
        self.assertEqual(_minimax_preset_mode("🔄 MiniMax H3 NSFW FL2VA (10s)"), "FL2VA")
        self.assertEqual(_minimax_preset_mode("🎞️ MiniMax H3 NSFW R2VA (5s)"), "R2VA")
        self.assertIsNone(_minimax_preset_mode("🎬 MiniMax H3 NSFW (5s)"))

    def test_match_minimax_preset_prefers_same_mode(self):
        preset_widget = {
            "options": {
                "values": [
                    "🎬 MiniMax H3 NSFW (5s)",
                    "🎬 MiniMax H3 NSFW (10s)",
                    "🔄 MiniMax H3 NSFW FL2VA (5s)",
                    "🔄 MiniMax H3 NSFW FL2VA (10s)",
                    "🎞️ MiniMax H3 NSFW R2VA (5s)",
                ]
            }
        }
        self.assertEqual(
            _match_minimax_preset(preset_widget, 10, "🔄 MiniMax H3 NSFW FL2VA (5s)"),
            "🔄 MiniMax H3 NSFW FL2VA (10s)",
        )
        self.assertEqual(
            _match_minimax_preset(preset_widget, 10, "🎬 MiniMax H3 NSFW (5s)"),
            "🎬 MiniMax H3 NSFW (10s)",
        )

    def test_minimax_result_keeps_fl2va_preset_family(self):
        graph = {
            "nodes": [
                {
                    "id": 105,
                    "widgets": [
                        {"name": "unet_name", "value": "x", "options": {"values": ["minimax_h3_fl2va_pruned_nvfp4_convrot_int8.safetensors"]}},
                        {
                            "name": "preset_prompt",
                            "value": "🔄 MiniMax H3 NSFW FL2VA (5s)",
                            "options": {
                                "values": [
                                    "🎬 MiniMax H3 NSFW (5s)",
                                    "🎬 MiniMax H3 NSFW (10s)",
                                    "🔄 MiniMax H3 NSFW FL2VA (5s)",
                                    "🔄 MiniMax H3 NSFW FL2VA (10s)",
                                ]
                            },
                        },
                        {"name": "passthrough", "value": True},
                        {"name": "prompt", "value": ""},
                        {"name": "steps", "value": 8, "options": {"values": ["res_multistep", "euler"]}},
                        {"name": "sampler_name", "value": "euler", "options": {"values": ["euler", "res_multistep"]}},
                        {"name": "scheduler", "value": "simple", "options": {"values": ["simple"]}},
                        {"name": "shift_video", "value": 6, "options": {"values": [6, 12]}},
                        {"name": "shift_audio", "value": 3, "options": {"values": [3]}},
                        {"name": "value_1", "value": 5},
                    ],
                }
            ]
        }
        result = _minimax_result(graph, "native", "use native 10 seconds")
        self.assertIsNotNone(result)
        preset_action = next(a for a in result["actions"] if a["widget"] == "preset_prompt")
        self.assertEqual(preset_action["value"], "🔄 MiniMax H3 NSFW FL2VA (10s)")
        value_action = next(a for a in result["actions"] if a["widget"] == "value_1")
        self.assertEqual(value_action["value"], 10)

    def test_image_enhancer_detected_by_image_input(self):
        node = {
            "id": 105,
            "type": "4c314f31-ecda-4b08-ae98-faaba1bf613f",
            "title": None,
            "widgets": [
                {"name": "prompt", "value": ""},
                {"name": "preset_prompt", "value": "🔄 MiniMax H3 NSFW FL2VA (5s)"},
                {"name": "passthrough", "value": True},
            ],
            "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],
        }
        self.assertTrue(_is_image_enhancer_node(node))
        self.assertTrue(_has_image_enhancer_target({"nodes": [node]}))

    def test_prompt_suppresses_video_guides_for_uuid_image_enhancer(self):
        graph = {
            "nodes": [
                {
                    "id": 105,
                    "type": "4c314f31-ecda-4b08-ae98-faaba1bf613f",
                    "title": None,
                    "widgets": [
                        {"name": "prompt", "value": ""},
                        {"name": "preset_prompt", "value": "🔄 MiniMax H3 NSFW FL2VA (5s)"},
                        {"name": "passthrough", "value": True},
                    ],
                    "inputs": [{"name": "image", "type": "IMAGE", "link": 1}],
                }
            ]
        }
        prompt = build_prompt([{"role": "user", "content": "10 seconds video"}], graph, has_images=True)
        self.assertIn("Exact image-enhancer target", prompt)
        self.assertNotIn("integrated_multimodal_description:", prompt)
        self.assertNotIn("overall_soundscape:", prompt)

    def test_fix_minimax_preset_corrects_generic_to_fl2va(self):
        graph = {
            "nodes": [
                {
                    "id": 105,
                    "widgets": [
                        {"name": "unet_name", "value": "x"},
                        {"name": "preset_prompt", "value": "🔄 MiniMax H3 NSFW FL2VA (5s)", "options": {"values": [
                            "🎬 MiniMax H3 NSFW (5s)",
                            "🎬 MiniMax H3 NSFW (10s)",
                            "🔄 MiniMax H3 NSFW FL2VA (5s)",
                            "🔄 MiniMax H3 NSFW FL2VA (10s)",
                        ]}},
                        {"name": "passthrough", "value": True},
                    ],
                }
            ]
        }
        result = {
            "message": "Done",
            "actions": [
                {"type": "set_widget_value", "node_id": 105, "widget": "preset_prompt", "value": "🎬 MiniMax H3 NSFW (10s)"},
                {"type": "queue_workflow"},
            ],
        }
        fixed = _fix_minimax_preset_actions(result, graph)
        preset_action = next(a for a in fixed["actions"] if a["widget"] == "preset_prompt")
        self.assertEqual(preset_action["value"], "🔄 MiniMax H3 NSFW FL2VA (10s)")


    def test_ensure_i2va_binding_prepends_missing_line(self):
        binding = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
        body = "integrated_multimodal_description: [Shot 1] A woman smiles."
        result = ensure_i2va_binding(body, "🎬 MiniMax H3 NSFW (5s)", has_image=True)
        self.assertTrue(result.startswith(binding))
        self.assertIn(body, result)

    def test_ensure_i2va_binding_keeps_existing_line(self):
        binding = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
        original = f"{binding}\n\nintegrated_multimodal_description: [Shot 1] A woman smiles."
        self.assertEqual(ensure_i2va_binding(original, "🎬 MiniMax H3 NSFW (5s)", has_image=True), original)

    def test_ensure_i2va_binding_skips_fl2va(self):
        body = "How the reference pictures align with the target video ..."
        self.assertEqual(
            ensure_i2va_binding(body, "🔄 MiniMax H3 NSFW FL2VA (5s)", has_image=True),
            body,
        )

    def test_normalize_minimax_removes_duplicate_shot_blocks(self):
        binding = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
        body = "integrated_multimodal_description: [Shot 1] ...\noverall_soundscape: ...\nnon_diegetic_music: N/A"
        duplicate = f"[Shot 1] A woman smiles.\n[Shot 2] She turns.\n\n{binding}\n\n{body}"
        result = normalize_minimax_output(duplicate, "🎬 MiniMax H3 NSFW (5s)", has_image=True)
        self.assertTrue(result.startswith(binding))
        self.assertNotIn("[Shot 2] She turns.", result.split("integrated_multimodal_description:")[0])
        self.assertIn("integrated_multimodal_description:", result)

    def test_normalize_minimax_adds_missing_i2va_binding(self):
        body = "integrated_multimodal_description: [Shot 1] A woman smiles.\noverall_soundscape: ...\nnon_diegetic_music: N/A"
        result = normalize_minimax_output(body, "🎬 MiniMax H3 NSFW (5s)", has_image=True)
        self.assertIn("For the target video", result)
        self.assertIn("integrated_multimodal_description:", result)

    def test_normalize_minimax_replaces_noncanonical_i2va_binding(self):
        malformed = "For the target video, at 0.00 seconds into the target video, Picture 1 (from [Shot 1]) is fully referenced."
        body = "integrated_multimodal_description: [Shot 1] A woman smiles.\noverall_soundscape: ...\nnon_diegetic_music: N/A"
        result = normalize_minimax_output(f"{malformed}\n\n{body}", "MiniMax › NSFW", has_image=True)
        self.assertEqual(result.count("For the target video"), 1)
        self.assertTrue(result.startswith(MINIMAX_I2VA_BINDING))
        self.assertNotIn(malformed, result)

    def test_normalize_minimax_keeps_fl2va_alignment(self):
        alignment = "How the reference pictures align with the target video — Picture 1 (from [Shot 1]) aligns with the 0.00-second mark; Picture 2 aligns with the 5.00-second mark."
        body = "integrated_multimodal_description: [Shot 1] ...\noverall_soundscape: ...\nnon_diegetic_music: N/A"
        text = f"{alignment}\n\n{body}"
        result = normalize_minimax_output(text, "🔄 MiniMax H3 NSFW FL2VA (5s)", has_image=True)
        self.assertTrue(result.startswith(alignment))
        self.assertIn("integrated_multimodal_description:", result)

    def test_normalize_minimax_synthesizes_missing_fl2va_alignment(self):
        body = "integrated_multimodal_description: [Shot 1] ...\noverall_soundscape: ...\nnon_diegetic_music: N/A"
        result = normalize_minimax_output(body, "🔄 MiniMax H3 NSFW FL2VA (5s)", has_image=True, duration="10s")
        self.assertTrue(result.startswith("How the reference pictures align"))
        self.assertIn("10.00-second mark", result)
        self.assertIn("integrated_multimodal_description:", result)

    def test_accepts_agent_tool_actions(self):
        actions = validate_actions([
            {"type": "search_models", "query": "face lora", "types": "LORA"},
            {"type": "download_model", "url": "https://civitai.com/api/download/models/1", "kind": "lora"},
            {"type": "clear_graph"},
        ])
        self.assertEqual(len(actions), 3)
        self.assertEqual(actions[0]["query"], "face lora")
        self.assertEqual(actions[1]["kind"], "lora")

    def test_rejects_agent_tool_misuse(self):
        self.assertEqual(validate_actions([
            {"type": "search_models", "query": ""},
            {"type": "download_model", "url": "https://civitai.com/api/download/models/1", "kind": "spaceship"},
        ]), [])
        # Host gating is enforced at download time, not shape time.
        import chat_service as cs
        with self.assertRaises(ValueError):
            cs.download_model("https://evil.example/x.safetensors", "lora")

    def test_civitai_search_shaping(self):
        import json as _json
        from unittest import mock
        import chat_service as cs

        payload = {"items": [{
            "name": "Face Fix", "type": "LORA",
            "modelVersions": [
                {"id": 11, "name": "v1", "baseModel": "SDXL",
                 "files": [{"downloadUrl": "https://civitai.com/api/download/models/11", "sizeKB": 204800}]},
                {"id": 12, "name": "broken", "baseModel": "SDXL", "files": []},
            ],
        }]}

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return _json.dumps(payload).encode()

        with mock.patch.object(cs.urllib.request, "urlopen", return_value=_Resp()):
            items = cs.civitai_search("face", "", 8)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["versions"][0]["url"], "https://civitai.com/api/download/models/11")
        self.assertEqual(len(items[0]["versions"]), 1)

    def test_installed_models_context_never_raises(self):
        import chat_service as cs
        cs._models_context_cache.update(at=0.0, text="")
        text = cs.installed_models_context()
        self.assertIsInstance(text, str)
        for folder in cs.MODELS_CONTEXT_FOLDERS:
            self.assertIn(folder, text)

    def test_accepts_queue_and_workflow_actions(self):
        actions = validate_actions([
            {"type": "queue_status"},
            {"type": "queue_cancel", "scope": "pending", "prompt_id": "abc123"},
            {"type": "queue_cancel", "scope": "all"},
            {"type": "workflow_list"},
            {"type": "workflow_save", "name": "CUSTOM portrait v2.json"},
            {"type": "workflow_load", "name": "CUSTOM_art-krea.json"},
            {"type": "workflow_delete", "name": "old test"},
        ])
        self.assertEqual(len(actions), 7)
        self.assertEqual(actions[1]["prompt_id"], "abc123")
        self.assertEqual(actions[2]["scope"], "all")
        self.assertEqual(actions[4]["name"], "CUSTOM portrait v2")
        self.assertEqual(actions[5]["name"], "CUSTOM_art-krea")
        self.assertEqual(actions[6]["name"], "old test")

    def test_rejects_queue_and_workflow_misuse(self):
        self.assertEqual(validate_actions([
            {"type": "queue_cancel", "scope": "everything"},
            {"type": "workflow_save", "name": "a/b"},
            {"type": "workflow_save", "name": "../evil"},
            {"type": "workflow_save", "name": "x" * 101},
            {"type": "workflow_load", "name": 5},
            {"type": "workflow_delete", "name": ".json"},
        ]), [])

    def test_workflow_name_normalization(self):
        import chat_service as cs
        self.assertEqual(cs._workflow_name("  My Workflow.json  "), "My Workflow")
        self.assertEqual(cs._workflow_name("plain"), "plain")
        self.assertEqual(cs._workflow_name("a\\b"), "")
        self.assertEqual(cs._workflow_name("a\tb"), "")
        self.assertEqual(cs._workflow_name(""), "")
        self.assertEqual(cs._workflow_name(None), "")

    @staticmethod
    def _comfy_router(running_body=b'{"running": []}', stats_body=None, stats_error=False, calls=None):
        """URL-routing fake for urllib in the _comfy_free tests."""
        class _Resp:
            def __init__(self, body=b""):
                self._body = body
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return self._body

        def route(request, timeout=None):
            url = getattr(request, "full_url", str(request))
            if calls is not None:
                calls.append(url)
            if url.endswith("/running"):
                return _Resp(running_body)
            if url.endswith("/system_stats"):
                if stats_error:
                    raise OSError("stats unavailable")
                return _Resp(stats_body)
            return _Resp()  # POST /free

        return route

    def test_comfy_free_refuses_occupied_card(self):
        import chat_service as cs
        from unittest import mock

        stats = b'{"devices": [{"name": "cuda:0", "vram_free": 5000000000, "vram_total": 25000000000}]}'
        with mock.patch.object(cs.urllib.request, "urlopen",
                               side_effect=self._comfy_router(stats_body=stats)), \
                mock.patch.object(cs, "FREE_WAIT_S", 0.0):
            with self.assertRaises(ValueError):
                cs._comfy_free("NInfer-HauHauCS-27B")

    def test_comfy_free_passes_when_card_is_free(self):
        import chat_service as cs
        from unittest import mock

        stats = b'{"devices": [{"name": "cuda:0", "vram_free": 23000000000, "vram_total": 25000000000}]}'
        with mock.patch.object(cs.urllib.request, "urlopen",
                               side_effect=self._comfy_router(stats_body=stats)), \
                mock.patch.object(cs, "FREE_WAIT_S", 0.0):
            cs._comfy_free("NInfer-HauHauCS-27B")

    def test_comfy_free_fails_open_when_stats_unreadable(self):
        import chat_service as cs
        from unittest import mock

        with mock.patch.object(cs.urllib.request, "urlopen",
                               side_effect=self._comfy_router(stats_error=True)), \
                mock.patch.object(cs, "FREE_WAIT_S", 0.0):
            cs._comfy_free("NInfer-HauHauCS-27B")  # must not raise

    def test_comfy_free_skips_when_model_already_warm(self):
        import chat_service as cs
        from unittest import mock

        warm = b'{"running": [{"model": "NInfer-HauHauCS-27B", "state": "ready"}]}'
        calls = []
        with mock.patch.object(cs.urllib.request, "urlopen",
                               side_effect=self._comfy_router(running_body=warm, calls=calls)):
            cs._comfy_free("NInfer-HauHauCS-27B")
        # No /free, no stats: a warm model must not be shoved aside.
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0].endswith("/running"))

    def test_comfy_free_allows_llamaswap_swap(self):
        import chat_service as cs
        from unittest import mock

        other = b'{"running": [{"model": "agent", "state": "ready"}]}'
        stats = b'{"devices": [{"name": "cuda:0", "vram_free": 5000000000, "vram_total": 25000000000}]}'
        with mock.patch.object(cs.urllib.request, "urlopen",
                               side_effect=self._comfy_router(running_body=other, stats_body=stats)), \
                mock.patch.object(cs, "FREE_WAIT_S", 0.0):
            cs._comfy_free("NInfer-HauHauCS-27B")  # llama-swap will swap; no raise

    def _runtime_with_scripted_api(self, responses):
        """ChatRuntime whose API backend returns scripted replies."""
        runtime = ChatRuntime()
        runtime.models = lambda: {"api": ["NInfer-HauHauCS-27B"], "hf": [], "gguf": []}
        calls = []

        def fake_generate(backend, model_name, prompt, options, enable_thinking, images, video):
            calls.append((backend, model_name, enable_thinking))
            return responses.pop(0), ""

        runtime._generate = fake_generate
        return runtime, calls

    def test_chat_parses_on_first_attempt(self):
        runtime, calls = self._runtime_with_scripted_api([
            '{"message":"steps set","actions":[{"type":"set_widget_value","node_id":1,"widget":"steps","value":20}],"choices":[]}',
        ])
        result = runtime.chat("api", "NInfer-HauHauCS-27B",
                              [{"role": "user", "content": "set 20 steps"}], {"nodes": []}, {})
        self.assertTrue(result["parsed"])
        self.assertEqual(result["message"], "steps set")
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(result["actions"][0]["widget"], "steps")
        self.assertEqual(len(calls), 1)

    def test_chat_retries_until_model_emits_json(self):
        runtime, calls = self._runtime_with_scripted_api([
            "We need answer user's request: set steps. Need inspect snapshot and identify errors.",
            "Still no JSON here, just analysis of the four branches.",
            '{"message":"ok","actions":[{"type":"set_widget_value","node_id":1,"widget":"steps","value":20}],"choices":[]}',
        ])
        result = runtime.chat("api", "NInfer-HauHauCS-27B",
                              [{"role": "user", "content": "fix the workflow"}], {"nodes": []}, {})
        self.assertTrue(result["parsed"])
        self.assertEqual(len(result["actions"]), 1)
        self.assertEqual(len(calls), 3)

    def test_chat_exhausts_json_attempts_returns_no_actions(self):
        runtime, calls = self._runtime_with_scripted_api([
            "Rambling one.",
            "Rambling two.",
            "Rambling three.",
        ])
        result = runtime.chat("api", "NInfer-HauHauCS-27B",
                              [{"role": "user", "content": "fix the workflow"}], {"nodes": []}, {})
        self.assertFalse(result["parsed"])
        self.assertEqual(result["actions"], [])
        self.assertEqual(len(calls), 3)

if __name__ == "__main__":
    unittest.main()

