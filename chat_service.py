import base64
import gc
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

from qwenvl_presets import VL_ALIASES, resolve_preset, resolve_vl_preset


MAX_MESSAGES = 20
MAX_MESSAGE_CHARS = 48000
# Total history budget (~100-130K tokens): NInfer-HauHauCS-27B has 262144
# ctx, shared with the workflow snapshot, guides, images and the reply.
# Oldest messages drop first; the latest user message is never dropped.
PROMPT_BUDGET_CHARS = 400000
# Workflow snapshot cap (~35-50K tokens). Values inside are already
# truncated by validate_graph; this bounds pathological graphs.
GRAPH_SNAPSHOT_CHARS = 150000
MAX_GRAPH_NODES = 200
MAX_CHAT_IMAGES = 3
MAX_VIDEO_FRAMES = 4
MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_ACTIONS = {"set_widget_value", "set_node_mode", "queue_workflow", "add_node", "remove_node", "connect", "disconnect", "layout", "search_models", "download_model", "clear_graph", "queue_status", "queue_cancel", "workflow_list", "workflow_save", "workflow_load", "workflow_delete"}

# Workflow-library names: the panel appends ".json" in the userdata workflows
# dir, so a name must stay a plain filename — no separators, no escapes.
WORKFLOW_NAME_CHARS = set("/\\:*?\"<>|%")
WORKFLOW_NAME_MAX = 100

# Model-store kinds the download action may write to (folder_paths keys).
DOWNLOAD_KINDS = {"checkpoint", "lora", "vae", "embedding", "controlnet"}
KIND_FOLDERS = {
    "checkpoint": ("checkpoints", (".safetensors", ".ckpt", ".pt", ".bin")),
    "lora": ("loras", (".safetensors", ".pt", ".bin")),
    "vae": ("vae", (".safetensors", ".pt", ".bin", ".ckpt")),
    "embedding": ("embeddings", (".safetensors", ".pt", ".bin")),
    "controlnet": ("controlnet", (".safetensors", ".pt", ".bin", ".ckpt")),
}

# civitai.com download URLs redirect to a CDN: the initial host is checked,
# redirects are followed, the payload lands only in a KIND_FOLDERS dir.
DOWNLOAD_HOSTS = {"civitai.com", "www.civitai.com", "api.civitai.com"}

# Installed-models context: scanned per chat (cached), capped, so the model
# picks real filenames instead of hallucinating them.
MODELS_CONTEXT_FOLDERS = ("checkpoints", "loras", "vae", "embeddings", "controlnet")
MODELS_CONTEXT_FILES = 60
MODELS_CONTEXT_TTL_S = 120
_models_context_cache = {"at": 0.0, "text": ""}

# API backend (llama-swap, OpenAI-compatible). Lets QwenChat's workflow-control
# protocol run on a served model instead of a locally loaded GGUF/HF weight —
# no second 27B in ComfyUI's VRAM. Endpoint and model are overridable via env
# for testing; defaults target aphotic's llama-swap + NInfer-HauHauCS-27B.
API_BASE_URL = os.environ.get("QWENCHAT_API_BASE_URL", "http://127.0.0.1:9090/v1")
API_DEFAULT_MODEL = os.environ.get("QWENCHAT_API_MODEL", "NInfer-HauHauCS-27B")
# Generous: covers queue-wait aftermath, NInfer cold load (~50s), big-prompt
# prefill and long generations. The browser side opts out of ComfyUI
# frontend's fetch timeout for this route (timeoutMs: null in chat.js).
API_TIMEOUT_S = int(os.environ.get("QWENCHAT_API_TIMEOUT_S", "600"))

# Ping-pong: the api chat model (llama-swap) and diffusion weights share one
# card. Before asking llama-swap to load, yield to any running ComfyUI job
# (wait at the job boundary, never cutting one off) and then unload Comfy
# models via POST /free — the mirror of the queue path, which unloads the
# chat model before a workflow runs. Without this the load collides with
# resident weights and llama-swap 500s ("upstream command exited prematurely").
COMFY_URL = os.environ.get("QWENCHAT_COMFY_URL", "http://127.0.0.1:8188")
# llama-swap's management API (no /v1 suffix): /running tells us whether the
# chat model is already up — a warm model occupies the card itself, so the
# free-VRAM guard must not run against it.
LLAMA_SWAP_URL = os.environ.get("LLAMA_SWAP_URL", "http://127.0.0.1:9090")
WAIT_FOR_IDLE = os.environ.get("QWENCHAT_WAIT_FOR_IDLE", "1") == "1"
QUEUE_POLL_S = float(os.environ.get("QWENCHAT_QUEUE_POLL_S", "5"))
# /free flags are applied asynchronously by ComfyUI's queue worker, so poll
# /system_stats after the call: wait up to FREE_WAIT_S for the card to free.
# The served chat model (NInfer-HauHauCS-27B) fills ~22.6 GiB of the 4090 and
# the healthy post-/free state leaves ~23 GiB free, so requiring most of the
# card separates "weights released" from "still resident" by a wide margin.
FREE_POLL_S = float(os.environ.get("QWENCHAT_FREE_POLL_S", "1.5"))
FREE_WAIT_S = float(os.environ.get("QWENCHAT_FREE_WAIT_S", "30"))
FREE_VRAM_MIN_FRACTION = float(os.environ.get("QWENCHAT_MIN_FREE_VRAM_FRACTION", "0.90"))
MINIMAX_I2VA_BINDING = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."


try:
    from wildcard_util import expand_wildcard_tokens as _expand_wildcard_tokens
except Exception:
    def _expand_wildcard_tokens(text):
        return text


def _is_minimax_i2va(preset_name, has_image=False):
    name = str(preset_name or "").lower()
    return has_image and "minimax" in name and "r2va" not in name and "fl2va" not in name


def ensure_i2va_binding(text, preset_name, has_image=False):
    """Ensure MiniMax H3 I2VA outputs include the required reference binding
    line. FL2VA/R2VA presets use their own alignment format and are excluded."""
    if not _is_minimax_i2va(preset_name, has_image):
        return text
    if MINIMAX_I2VA_BINDING in text:
        return text
    return f"{MINIMAX_I2VA_BINDING}\n\n{text.lstrip()}"


def _fl2va_alignment_line(text, duration):
    """Synthesize the FL2VA reference-alignment header when the model dropped it."""
    seconds = None
    if duration:
        match = re.search(r"(\d+(?:\.\d+)?)", str(duration))
        if match:
            seconds = float(match.group(1))
    if seconds is None:
        marks = [float(v) for v in re.findall(r"(\d+(?:\.\d+)?)\s*-?\s*second mark", text)]
        if marks:
            seconds = max(marks)
    if seconds is None:
        return None
    return (
        "How the reference pictures align with the target video — "
        "Picture 1 (from [Shot 1]) aligns with the 0.00-second mark of the target video; "
        f"Picture 2 (from [Shot 1]) aligns with the {seconds:.2f}-second mark of the target video."
    )


def normalize_minimax_output(text, preset_name, has_image=False, duration=None):
    """Remove stray preface / duplicated shot blocks before the real prompt body
    and ensure the required reference alignment line is present.

    MiniMax H3 presets expect the output to start directly with the mode's
    header (I2VA binding / FL2VA alignment line / nothing for T2VA), followed
    by `integrated_multimodal_description:`, `overall_soundscape:` and
    `non_diegetic_music:`. Anything else before the first body label is noise."""
    if not preset_name or "MiniMax" not in preset_name:
        return text
    label = "integrated_multimodal_description:"
    idx = text.find(label)
    if idx < 0:
        return ensure_i2va_binding(text, preset_name, has_image)

    head, tail = text[:idx], text[idx:]
    marker = None
    if "FL2VA" in preset_name or "R2VA" in preset_name:
        marker = "reference pictures"
    elif has_image:
        marker = "For the target video"

    prefix = ""
    if _is_minimax_i2va(preset_name, has_image):
        prefix = MINIMAX_I2VA_BINDING + "\n\n"
    elif marker:
        for line in head.splitlines():
            if marker.lower() in line.lower():
                prefix = line.strip() + "\n\n"
                break
    if not prefix and "FL2VA" in preset_name:
        synthesized = _fl2va_alignment_line(text, duration)
        if synthesized:
            prefix = synthesized + "\n\n"
    return ensure_i2va_binding(prefix + tail.lstrip(), preset_name, has_image)


BASE_SYSTEM_PROMPT = """You are Qwen Workflow Assistant inside ComfyUI. CRITICAL: your entire response must be exactly one JSON object — no text before or after it, no think tags, no analysis, no explanation. If you output anything besides JSON, the workflow will not be modified. Answer the user via the "message" field and, only when requested, control the open workflow using the supplied snapshot.
LANGUAGE: "message" and choice labels must mirror the LATEST user message language. Workflow prompt text must be English.
OUTPUT: return exactly one JSON object, no text outside it:
{"message":"reply","actions":[{"type":"set_widget_value","node_id":1,"widget":"steps","value":25},{"type":"queue_workflow"}],"choices":[{"label":"A","send":"text for A"}]}
Example: to fix scrambled KSampler widgets:
{"message":"Fixed KSampler 12: steps 20, cfg 1, sampler euler, scheduler simple, control_after_generate fixed","actions":[{"type":"set_widget_value","node_id":12,"widget":"steps","value":20},{"type":"set_widget_value","node_id":12,"widget":"cfg","value":1},{"type":"set_widget_value","node_id":12,"widget":"sampler_name","value":"euler"},{"type":"set_widget_value","node_id":12,"widget":"scheduler","value":"simple"},{"type":"set_widget_value","node_id":12,"widget":"control_after_generate","value":"fixed"}],"choices":[]}
Your response must START with "{" and END with "}". Do not analyze, explain, or describe anything before or after the JSON object.
Allowed actions: set_widget_value, set_node_mode (bypass/enable only), queue_workflow, add_node, remove_node, connect, disconnect, layout, search_models, download_model, clear_graph, queue_status, queue_cancel, workflow_list, workflow_save, workflow_load, workflow_delete. Never invent numeric node IDs or widget names — nodes you create are referenced by the key you assign them (see 4). Never emit code, shell, filesystem, JS, or network actions.
When asked to generate/run/queue, apply every required widget update and include queue_workflow in the same response. Never ask "execute now?" after queue_workflow. If you only set a prompt without queue_workflow, do not claim execution started.
When asked to fix or resolve errors in the workflow, emit the correcting actions (set_widget_value, connect, disconnect, ...) and include queue_workflow when the user wants it run — describing the errors is not enough.
When a fix involves multiple nodes or widgets, emit ONE action per response (e.g. one set_widget_value). The panel applies it and you can continue with the next fix in the following round. Do not try to fix everything at once.
Choices are top-level, only when truly ambiguous. Always close braces.
---
WORKFLOW TARGETS (pick the FIRST matching case for the node you are controlling):
1. MiniMax H3 video sampler (exposes unet_name + preset_prompt + passthrough):
   - If the request says "use Native", "use 10Eros Turbo", "Config A" etc., FIRST update the sampler widgets, then write only the action into the "prompt" widget.
   - Config mapping:
     * Native / Config C uses the minimax_h3_fl2va_pruned_nvfp4_convrot_int8 unet with 20 steps, res_multistep sampler, simple scheduler, shift_video 12, shift_audio 3
     * Native Turbo / Config B uses the same unet with 8 steps, euler sampler, simple scheduler, shift_video 6, shift_audio 3
     * 10Eros / Config D uses the 10Eros_Max_h3_hybrid_beta5_int8 unet with 20 steps, res_multistep sampler, simple scheduler, shift_video 12, shift_audio 3
     * 10Eros Turbo / Config A uses the same 10Eros unet with 8 steps, euler sampler, simple scheduler, shift_video 6, shift_audio 3
   - Turbo variants run with the turbo LoRA enabled; the LoRA node toggle is handled automatically, do not mention it.
   - If a different duration is requested, set the "value_1" widget to that number of seconds.
   - For the prompt: remove config words and duration. Write only a short English action description.
   - EXAMPLE: user says "generate a 5s video, use Native: rhythmic hip sway, subtle back and forth"
     You set value_1 to 5 on the sampler, select the minimax unet, set 20 steps with res_multistep, set shift_video 12 and shift_audio 3, write "rhythmic hip sway, subtle back and forth" into the prompt widget, set passthrough to false, and queue the workflow.
   - EXAMPLE: user says "use 10Eros Turbo: slow caressing on thigh, static camera"
     You select the 10Eros unet, set 8 steps with euler, set shift_video 6, write "slow caressing on thigh, static camera" into the prompt widget, set passthrough to false, and queue the workflow.
   - NEVER copy "use Native", "use 10Eros", "generate", "5s video" into the prompt widget.
   - NEVER describe the image yourself (clothes, face, room, light); the inner QwenVL model will see the image and describe it. You only provide the action.
2. Passthrough node (has passthrough but NO image pixels): write the complete final English prompt into the actual prompt widget and set passthrough=true.
3. Direct prompt: write the complete final English prompt into the generation widget. Use attached images as context when present.
When you set a text/prompt widget, repeat the new value verbatim in "message" so the user sees it.
Only write prompts into real generation widgets (prompt/custom_prompt/prompt_text). NEVER write into display nodes like ShowText, easy showAnything, or MarkdownNote.
4. Structural edits (live graph surgery — the snapshot now carries inputs, outputs, links and positions):
   - Add: {"type":"add_node","key":"k1","node_type":"KSampler"} with your own short key; later actions in the SAME response use "k1" and the panel maps it to the real id. Order matters: add first, then set its widgets, then connect it. Omit x/y for auto-place. Only use node types that exist in ComfyUI (types already in the snapshot are safe; standard ones like KSampler, CheckpointLoaderSimple, CLIPTextEncode, VAEDecode, EmptyLatentImage, PreviewImage, LoraLoader are fine). If creation fails the panel reports why — ask the user, don't retry blindly.
   - Wire: {"type":"connect","from":id_or_key,"out":"MODEL","to":id,"in":"model"} using slot names. A target input that is already linked must be freed first: {"type":"disconnect","node_id":id,"input":"model"}.
   - Remove: {"type":"remove_node","node_id":id} — only nodes the user asked to remove; removing a node deletes its links.
   - Tidy: {"type":"layout"} rearranges the whole graph into neat columns ("organize this workflow"). Report structural results (new ids, links made/broken) in "message".
5. Live assistant (multi-round: you act, the panel reports back, you continue):
   - You may work over several rounds in one response cycle: tool results and rejected actions come back as user messages prefixed OBSERVATION — read them and continue (fix, retry with other values, or finish). Stop when done; never queue twice.
   - Building from a blank canvas: chain loaders → encoders → sampler → decode → output (e.g. CheckpointLoaderSimple → CLIPTextEncode ×2 → EmptyLatentImage → KSampler → VAEDecode → PreviewImage/SaveImage), set each widget, connect by slot names, layout, then queue. If the checkpoint is uncertain, ask via choices instead of guessing.
   - Model store: the INSTALLED MODELS block lists real filenames — always prefer them in loader widgets. If the user names something missing, search CivitAI: {"type":"search_models","query":"...","types":"LORA"} and offer the hits as choices (label + send "download <name>: <url>"). Download only what the user picks: {"type":"download_model","url":"...","kind":"lora"} (kind: checkpoint/lora/vae/embedding/controlnet). The file lands in the store and is immediately usable by name.
   - clear_graph wipes the whole canvas: ONLY when the user explicitly says so ("start over", "clear", "new workflow"). Never as a side effect.
   - Queue control: {"type":"queue_status"} reports what is running/pending; {"type":"queue_cancel","scope":"running"|"pending"|"all"} stops work (add "prompt_id":"..." with scope pending to cancel one specific job). Cancel only when the user asks to stop something.
   - Workflow library (the saved workflows in the sidebar): {"type":"workflow_list"} lists saved names; {"type":"workflow_save","name":"..."} saves the open graph under that name — it never overwrites (a collision becomes name-2, name-3, ...; ".json" optional); {"type":"workflow_load","name":"..."} opens a saved workflow and REPLACES the canvas — use it when the user asks; if their canvas looks unfinished you may warn them first, and any further actions in the same response run on the loaded graph; {"type":"workflow_delete","name":"..."} permanently removes a saved workflow — ONLY when the user explicitly asks to delete that specific one."""

_LT = chr(60)
_GT = chr(62)
THINK_OPEN = _LT + "think" + _GT
THINK_CLOSE = _LT + "/think" + _GT
THINKING_INSTRUCTION = " Put your step-by-step reasoning inside " + THINK_OPEN + "..." + THINK_CLOSE + " tags, then return the JSON object after the closing " + THINK_CLOSE + " tag."
NO_THINKING_INSTRUCTION = " Do not output reasoning tags. Your response must be exactly one JSON object starting with { and ending with }. No prose, no analysis, no explanation, no markdown fences. Output format example: {\"message\":\"Fixed 4 KSampler widgets\",\"actions\":[{\"type\":\"set_widget_value\",\"node_id\":12,\"widget\":\"steps\",\"value\":20}],\"choices\":[]}."

SYSTEM_PROMPT = BASE_SYSTEM_PROMPT + NO_THINKING_INSTRUCTION


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("invalid chat message content")
    result = []
    for item in messages[-MAX_MESSAGES:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            raise ValueError("invalid chat message content")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("invalid chat message content")
        content = content.strip()
        if len(content) > MAX_MESSAGE_CHARS:
            # A long reply stored in history (up to max_tokens worth) must not
            # poison every later send with a 400 — truncate, head-kept.
            content = content[:MAX_MESSAGE_CHARS]
        result.append({"role": item["role"], "content": content})
    return result


def validate_graph(graph):
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ValueError("invalid workflow snapshot")
    nodes = graph["nodes"]
    if len(nodes) > MAX_GRAPH_NODES:
        raise ValueError(f"workflow exceeds {MAX_GRAPH_NODES} nodes")
    result = []
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), (int, str)):
            raise ValueError("invalid workflow node")
        widgets = node.get("widgets", [])
        if not isinstance(widgets, list):
            raise ValueError("invalid workflow widgets")
        safe_widgets = []
        for widget in widgets[:100]:
            if not isinstance(widget, dict) or not isinstance(widget.get("name"), str):
                continue
            value = widget.get("value")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                value = str(value)[:1000]
            options = widget.get("options") if isinstance(widget.get("options"), dict) else {}
            values = options.get("values")
            safe_widgets.append({
                "name": widget["name"][:200],
                "type": str(widget.get("type", ""))[:100],
                "value": value[:4000] if isinstance(value, str) else value,
                "options": {
                    "min": options.get("min") if isinstance(options.get("min"), (int, float)) else None,
                    "max": options.get("max") if isinstance(options.get("max"), (int, float)) else None,
                    "values": [str(item)[:500] for item in values[:200]] if isinstance(values, list) else None,
                },
            })
        safe_inputs = []
        for inp in node.get("inputs", []):
            if not isinstance(inp, dict) or not isinstance(inp.get("name"), str):
                continue
            link = inp.get("link")
            safe_inputs.append({
                "name": inp["name"][:200],
                "type": str(inp.get("type", ""))[:100],
                "link": link if isinstance(link, (int, type(None))) else None,
            })
        safe_outputs = []
        for out in node.get("outputs", []):
            if not isinstance(out, dict) or not isinstance(out.get("name"), str):
                continue
            links = out.get("links")
            safe_outputs.append({
                "name": out["name"][:200],
                "type": str(out.get("type", ""))[:100],
                "links": [link for link in links[:20] if isinstance(link, int)] if isinstance(links, list) else [],
            })
        pos = node.get("pos", [0, 0])
        safe_pos = [0, 0]
        if isinstance(pos, (list, tuple)) and len(pos) >= 2 and isinstance(pos[0], (int, float)) and isinstance(pos[1], (int, float)):
            safe_pos = [pos[0], pos[1]]
        result.append({
            "id": node["id"],
            "type": str(node.get("type", ""))[:200],
            "title": str(node.get("title", ""))[:200],
            "mode": node.get("mode", 0),
            "widgets": safe_widgets,
            "inputs": safe_inputs,
            "outputs": safe_outputs,
            "pos": safe_pos,
        })
    return {"nodes": result}


def _workflow_name(value):
    """Normalize a workflow-library name from the model: strip whitespace,
    drop a trailing .json, reject separators/controls/.. . Returns '' when
    unusable so the action is dropped before reaching the panel."""
    if not isinstance(value, str):
        return ""
    name = value.strip()
    if name.lower().endswith(".json"):
        name = name[:-5].strip()
    if not name or len(name) > WORKFLOW_NAME_MAX:
        return ""
    if any(ch in WORKFLOW_NAME_CHARS for ch in name) or ".." in name:
        return ""
    if any(not ch.isprintable() for ch in name):
        return ""
    return name


def validate_actions(actions):
    if not isinstance(actions, list):
        return []
    result = []
    for action in actions[:50]:
        if not isinstance(action, dict) or action.get("type") not in ALLOWED_ACTIONS:
            continue
        action_type = action["type"]
        if action_type == "queue_workflow":
            result.append({"type": action_type})
        elif action_type == "layout":
            result.append({"type": action_type})
        elif action_type == "set_node_mode" and isinstance(action.get("node_id"), (int, str)) and action.get("mode") in {"bypass", "enable"}:
            result.append({"type": action_type, "node_id": action["node_id"], "mode": action["mode"]})
        elif action_type == "set_widget_value" and isinstance(action.get("node_id"), (int, str)) and isinstance(action.get("widget"), str) and len(action["widget"]) <= 200:
            value = action.get("value")
            if value is None or isinstance(value, (str, int, float, bool)):
                if isinstance(value, str):
                    value = _expand_wildcard_tokens(value)
                result.append({"type": action_type, "node_id": action["node_id"], "widget": action["widget"], "value": value})
        elif action_type == "add_node" and isinstance(action.get("key"), str) and 0 < len(action["key"]) <= 80 and isinstance(action.get("node_type"), str) and 0 < len(action["node_type"]) <= 200:
            entry = {"type": action_type, "key": action["key"], "node_type": action["node_type"]}
            if isinstance(action.get("title"), str) and action["title"]:
                entry["title"] = action["title"][:200]
            for axis in ("x", "y"):
                if isinstance(action.get(axis), (int, float)):
                    entry[axis] = action[axis]
            result.append(entry)
        elif action_type == "remove_node" and isinstance(action.get("node_id"), (int, str)):
            result.append({"type": action_type, "node_id": action["node_id"]})
        elif action_type == "connect" and isinstance(action.get("from"), (int, str)) and isinstance(action.get("to"), (int, str)) and isinstance(action.get("out"), (str, int)) and isinstance(action.get("in"), (str, int)):
            result.append({"type": action_type, "from": action["from"], "to": action["to"], "out": action["out"], "in": action["in"]})
        elif action_type == "disconnect" and isinstance(action.get("node_id"), (int, str)) and isinstance(action.get("input"), (str, int)):
            result.append({"type": action_type, "node_id": action["node_id"], "input": action["input"]})
        elif action_type == "search_models" and isinstance(action.get("query"), str) and 0 < len(action["query"]) <= 300:
            entry = {"type": action_type, "query": action["query"]}
            if isinstance(action.get("types"), str) and action["types"]:
                entry["types"] = action["types"][:200]
            result.append(entry)
        elif action_type == "download_model" and isinstance(action.get("url"), str) and 0 < len(action["url"]) <= 2000 and isinstance(action.get("kind"), str) and action.get("kind") in DOWNLOAD_KINDS:
            entry = {"type": action_type, "url": action["url"], "kind": action["kind"]}
            if isinstance(action.get("filename"), str) and action["filename"]:
                entry["filename"] = action["filename"][:200]
            result.append(entry)
        elif action_type == "clear_graph":
            result.append({"type": action_type})
        elif action_type == "queue_status":
            result.append({"type": action_type})
        elif action_type == "queue_cancel":
            scope = action.get("scope")
            if scope in {"pending", "running", "all"}:
                entry = {"type": action_type, "scope": scope}
                prompt_id = action.get("prompt_id")
                if prompt_id is not None and isinstance(prompt_id, (int, str)) and 0 < len(str(prompt_id)) <= 64:
                    entry["prompt_id"] = prompt_id
                result.append(entry)
        elif action_type == "workflow_list":
            result.append({"type": action_type})
        elif action_type in {"workflow_save", "workflow_load", "workflow_delete"}:
            name = _workflow_name(action.get("name"))
            if name:
                result.append({"type": action_type, "name": name})
    return result


def validate_choices(choices):
    if not isinstance(choices, list):
        return []
    result = []
    for item in choices[:4]:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        send = item.get("send")
        if isinstance(label, str) and isinstance(send, str) and 0 < len(label) <= 80 and 0 < len(send) <= 500:
            result.append({"label": label.strip(), "send": send.strip()})
    return result


def list_output_images(output_dir, limit=500):
    root = Path(output_dir).resolve()
    if not root.is_dir():
        return []
    images = []
    for path in root.rglob("*"):
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mov"}:
                continue
            images.append((resolved.stat().st_mtime, resolved.relative_to(root).as_posix()))
        except (OSError, ValueError):
            continue
    images.sort(key=lambda item: item[0], reverse=True)
    return [f"{relative} [output]" for _, relative in images[:max(1, min(int(limit), 2000))]]


def validate_images(images, limit=MAX_CHAT_IMAGES):
    if not isinstance(images, list):
        return []
    result = []
    for item in images[:limit]:
        if not isinstance(item, str):
            continue
        try:
            data = base64.b64decode(item, validate=True)
        except (ValueError, TypeError):
            continue
        if len(data) > MAX_IMAGE_BYTES:
            continue
        result.append(data)
    return result


def _decode_images(image_data_list):
    result = []
    for data in image_data_list:
        try:
            image = Image.open(io.BytesIO(data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            result.append(image)
        except Exception:
            continue
    return result


def extract_thinking(text):
    text = (text or "").strip()
    thinking = ""
    pattern = re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE)
    match = re.search(pattern, text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        text = re.sub(pattern, "", text, count=1, flags=re.DOTALL).strip()
    return thinking, text


def parse_model_response(text):
    thinking, text = extract_thinking(text)
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1))
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first:last + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            message = data.get("message", "")
            return {
                "thinking": thinking,
                "message": message if isinstance(message, str) else str(message),
                "actions": validate_actions(data.get("actions", [])),
                "choices": validate_choices(data.get("choices", [])),
                "parsed": True,
            }
    message_fallback = text or "The model returned an empty response."
    # Salvage the message field from malformed JSON (e.g. unclosed brackets).
    salvage = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if salvage:
        try:
            message_fallback = json.loads('"' + salvage.group(1) + '"')
        except (TypeError, json.JSONDecodeError):
            pass
    return {"thinking": thinking, "message": message_fallback, "actions": [], "choices": [], "parsed": False}


_COMMAND_FILLERS = {
    "a", "adesso", "ahead", "avanti", "bene", "certo", "con", "coda", "di", "favore", "gia", "già",
    "grazie", "i", "il", "in", "it", "la", "le", "lo", "now", "ok", "okay", "ora", "per", "perfetto",
    "please", "precedente", "previous", "pure", "same", "si", "sì", "stesso", "subito", "sure",
    "thanks", "the", "this", "un", "una", "va", "with", "yes",
}
_COMMAND_OBJECTS = {"generation", "generazione", "job", "prompt", "queue", "task", "video", "workflow"}
_COMMAND_VERBS = {
    "avvia", "avvialo", "avviare", "cambia", "change", "conferma", "confermo", "confirm", "crea",
    "create", "do", "edit", "esegui", "eseguilo", "eseguire", "execute", "fallo", "genera",
    "generalo", "generate", "go", "lancia", "lancialo", "launch", "metti", "mettilo", "modifica",
    "modify", "procedi", "proceed", "queue", "run", "start", "vai",
}
_CONFIRM_ONLY = {"bene", "certo", "confermo", "grazie", "ok", "okay", "perfetto", "si", "sì", "sure", "va", "yes"}
_PREFORMATTED_PROMPT_MARKERS = (
    "integrated_multimodal_description", "overall_soundscape", "non_diegetic_music",
    "<picture", "for the target video", "[shot",
)


def _normalize_command(text):
    normalized = re.sub(r"[^\w\s]", "", (text or "").lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _is_execution_only(text):
    """True when the message is only a run/confirm command with no scene content."""
    words = _normalize_command(text).split()
    if not words or len(words) > 8:
        return False
    if all(word in _CONFIRM_ONLY for word in words):
        return True
    vocabulary = _COMMAND_FILLERS | _COMMAND_OBJECTS | _COMMAND_VERBS
    return any(word in _COMMAND_VERBS for word in words[:2]) and all(word in vocabulary for word in words)


def _is_enhancer_instruction(value):
    """True when a model-written prompt action is a usable action directive —
    not a bare confirmation and not a pre-formatted preset prompt."""
    if not isinstance(value, str) or not value.strip() or len(value) > 1500:
        return False
    lowered = value.lower()
    if any(marker in lowered for marker in _PREFORMATTED_PROMPT_MARKERS):
        return False
    return not _is_execution_only(value)


def select_workflow_intent(messages):
    intent = None
    last_user = None
    for item in messages or []:
        if item.get("role") != "user" or not isinstance(item.get("content"), str):
            continue
        content = item["content"].strip()
        if not content:
            continue
        last_user = content
        if not _is_execution_only(content):
            intent = content
    return intent if intent is not None else (last_user or "")


_CONFIG_PHRASE = re.compile(
    r"\b(?:use|using|with|in|switch(?:ing)?\s+to|usa|metti|passa\s+a)\s+"
    r"(?:the\s+)?(?:native|10\s*eros(?:-max)?|eros|turbo(?:\s*lora)?|config\s*[abc]|sol[-\s]?attn)"
    r"(?:\s+(?:mode|config|preset|model|version))?",
    re.IGNORECASE,
)
_GENERATION_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:generate|creates?|makes?|render|animate|produce|do|genera|crea|fai)\s+"
    r"(?:a\s+|an\s+|the\s+|this\s+|me\s+|one\s+|un\s+|una\s+|il\s+)?\s*"
    r"\d*\s*(?:sec(?:ond)?s?|s)?\s*"
    r"(?:second\s+|new\s+)?(?:video|clip|animation|scene)?\s*"
    r"(?:of|with|showing|where|di|con)?[:,]?\s*",
    re.IGNORECASE,
)
_DURATION_MENTION = re.compile(
    r"\b\d+\s*(?:s|sec(?:ond)?s?|secondi?)\s*(?:video|clip|animation)\b"
    r"|\b(?:video|clip|animation)\s+(?:of|di|da)\s+\d+\s*(?:s|sec(?:ond)?s?|secondi?)\b"
    r"|\b\d+s\b",
    re.IGNORECASE,
)
_LEADING_MEDIA = re.compile(
    r"^(?:(?:a|an|the|this|new|un|una|il|l)\s+)?(?:video|clip|animation|scene)\b[\s,;:.-]*",
    re.IGNORECASE,
)


_ACTION_DURATION = re.compile(r"(?:lasts?|lasting|at least|almeno|dura)\b\s*$", re.IGNORECASE)


def _clip_duration_seconds(text):
    """First duration mention that reads as a clip-length spec. Durations that
    describe the action itself ('the kiss lasts 5 seconds', 'at least 3
    seconds') are skipped so they never set the sampler's length widget."""
    if not isinstance(text, str):
        return None
    for match in re.finditer(r"\b(\d{1,2})\s*(?:sec(?:ond)?s?|s|secondi?)\b", text, re.IGNORECASE):
        if _ACTION_DURATION.search(text[max(0, match.start() - 30):match.start()]):
            continue
        return int(match.group(1))
    return None


def _clean_action_directive(text):
    """Strip routing keywords and generation meta from a directive so the
    enhancer only receives the scene action (e.g. 'use native' and
    'generate a 5s video' never reach the prompt widget). Returns an empty
    string when nothing but meta remains."""
    if not isinstance(text, str):
        return ""
    cleaned = _CONFIG_PHRASE.sub("", text).strip(" ,;:-").lstrip(".")
    cleaned = _GENERATION_PREFIX.sub("", cleaned, count=1).strip(" ,;:-").lstrip(".")
    cleaned = _GENERATION_PREFIX.sub("", cleaned, count=1).strip(" ,;:-").lstrip(".")
    cleaned = _DURATION_MENTION.sub("", cleaned)
    cleaned = re.sub(r"\s*[,;:]\s*", ", ", cleaned)
    cleaned = re.sub(r"(?:,\s*){2,}", ", ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).lstrip(" ,;:.-").rstrip(" ,;:-")
    cleaned = _LEADING_MEDIA.sub("", cleaned)
    cleaned = re.sub(r"^(?:(?:a|an|the)\s+)?(?:of|di|da|con|with|showing)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def enforce_image_enhancer_routing(result, graph, messages, has_images):
    if not has_images or not any(action.get("type") == "queue_workflow" for action in result.get("actions", [])):
        return result
    candidates = []
    for node in graph.get("nodes", []):
        widgets = {widget.get("name") for widget in node.get("widgets", []) if isinstance(widget, dict)}
        prompt_widget = next((name for name in ("prompt", "custom_prompt", "prompt_text") if name in widgets), None)
        if prompt_widget and {"preset_prompt", "passthrough"}.issubset(widgets) and _is_image_enhancer_node(node):
            candidates.append((node, prompt_widget))
    targeted = {
        str(action.get("node_id"))
        for action in result.get("actions", [])
        if action.get("type") == "set_widget_value" and action.get("widget") in {"prompt", "custom_prompt", "prompt_text", "preset_prompt", "passthrough"}
    }
    selected = [(node, widget) for node, widget in candidates if str(node.get("id")) in targeted]
    if len(selected) != 1:
        if len(candidates) != 1:
            return result
        selected = candidates
    node, prompt_widget = selected[0]
    node_id = node.get("id")
    intent = _clean_action_directive(select_workflow_intent(messages))
    actions = result.get("actions", [])
    prompt_action = next((action for action in actions if str(action.get("node_id")) == str(node_id) and action.get("widget") in {"prompt", "custom_prompt", "prompt_text"}), None)
    if prompt_action is not None and _is_enhancer_instruction(prompt_action.get("value")):
        intent = _clean_action_directive(prompt_action["value"].strip())
    if not intent:
        # Only config keywords (e.g. "use native") — leave the existing prompt
        if prompt_action is not None:
            actions.remove(prompt_action)
    else:
        if prompt_action is None:
            queue_index = next((index for index, action in enumerate(actions) if action.get("type") == "queue_workflow"), len(actions))
            prompt_action = {"type": "set_widget_value", "node_id": node_id}
            actions.insert(queue_index, prompt_action)
        prompt_action.update({"widget": prompt_widget, "value": intent})
    passthrough_action = next((action for action in actions if str(action.get("node_id")) == str(node_id) and action.get("widget") == "passthrough"), None)
    if passthrough_action:
        passthrough_action["value"] = False
    else:
        queue_index = next((index for index, action in enumerate(actions) if action.get("type") == "queue_workflow"), len(actions))
        actions.insert(queue_index, {"type": "set_widget_value", "node_id": node_id, "widget": "passthrough", "value": False})
    if intent:
        result["message"] = f'{result.get("message", "").rstrip()}\n\nWorkflow enhancer instruction:\n{intent}'.strip()
    return result


def enforce_image_reference_bindings(result, graph, has_images):
    if not has_images:
        return result
    nodes = {str(node.get("id")): node for node in graph.get("nodes", [])}
    # Use the passthrough value that the action set will write, not the snapshot's
    # old widget value, so a fresh passthrough=false does not get a binding line.
    passthrough_values = {
        str(action.get("node_id")): action.get("value")
        for action in result.get("actions", [])
        if action.get("type") == "set_widget_value" and action.get("widget") == "passthrough"
    }
    amended = []
    for action in result.get("actions", []):
        if action.get("type") != "set_widget_value" or action.get("widget") not in {"prompt", "custom_prompt", "prompt_text"}:
            continue
        value = action.get("value")
        node = nodes.get(str(action.get("node_id")))
        if not isinstance(value, str) or not node or MINIMAX_I2VA_BINDING in value:
            continue
        widgets = {widget.get("name"): widget.get("value") for widget in node.get("widgets", []) if isinstance(widget, dict)}
        preset = str(widgets.get("preset_prompt", ""))
        resolved_preset, _ = resolve_preset(preset, VL_ALIASES)
        is_minimax_i2va = (
            "minimax" in resolved_preset.lower()
            and "r2va" not in resolved_preset.lower()
            and "fl2va" not in resolved_preset.lower()
        )
        title = f'{node.get("title", "")} {node.get("type", "")}'.lower()
        passthrough = passthrough_values.get(str(action.get("node_id")), widgets.get("passthrough") is True)
        if not is_minimax_i2va or "image to video" not in title or not passthrough:
            continue
        action["value"] = f"{MINIMAX_I2VA_BINDING}\n\n{value.lstrip()}"
        amended.append(action["value"])
    if amended:
        result["message"] = f'{result.get("message", "").rstrip()}\n\nFinal prompt sent to workflow:\n{amended[-1]}'.strip()
    return result


def _node_input_names(node):
    return {str(inp.get("name", "")).lower() for inp in node.get("inputs", []) if isinstance(inp, dict)}


def _is_image_enhancer_node(node):
    """True if the node exposes a preset enhancer with an image/video input."""
    widgets = {widget.get("name"): widget for widget in node.get("widgets", []) if isinstance(widget, dict)}
    prompt_widget = next((name for name in ("prompt", "custom_prompt", "prompt_text") if name in widgets), None)
    if not prompt_widget or "preset_prompt" not in widgets or "passthrough" not in widgets:
        return False
    title = f'{node.get("title", "")} {node.get("type", "")}'.lower()
    input_names = _node_input_names(node)
    return "image to video" in title or bool({"image", "image2", "video"} & input_names)


def _has_image_enhancer_target(graph):
    """True if the workflow exposes an image-to-video node that has a preset
    enhancer (preset_prompt + passthrough + a prompt widget). In that case the
    chat must not write a fully formatted video prompt itself."""
    for node in graph.get("nodes", []):
        if _is_image_enhancer_node(node):
            return True
    return False


def _preset_guides(graph, messages, has_images=False):
    """Collect prompt-writing guides for presets selected in the workflow's
    widgets or named in the last user message.

    When an image-enhancer target exists, the chat is only the first stage of a
    two-stage pipeline: it must write a concise action directive, while the
    inner QwenVL node applies the preset and formats the final prompt. Full
    format guides for video presets are therefore suppressed to avoid confusing
    the chat into outputting a complete MiniMax/LTX/Wan prompt."""
    module = sys.modules.get("AILab_QwenVL")
    guides = getattr(module, "SYSTEM_PROMPTS", None) or {}
    if not guides:
        return ""
    wanted = set()
    for node in graph.get("nodes", []):
        widgets = {w.get("name"): w.get("value") for w in node.get("widgets", []) if isinstance(w, dict)}
        for widget in node.get("widgets", []):
            if not isinstance(widget, dict):
                continue
            value = widget.get("value")
            if value in guides:
                wanted.add(value)
            elif isinstance(value, str):
                # Widget values are base preset names; resolve to the flat
                # "Name (Ns)" guide key using the node's duration widget.
                _, key = resolve_vl_preset(value, widgets.get("duration"))
                if key in guides:
                    wanted.add(key)
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    wanted.update(name for name in guides if name in last_user)
    # A duration mention (e.g. "10 seconds", "10s", "10 secondi") selects the
    # flat guide for that length, e.g. "MiniMax › NSFW (10s)". Keep the
    # same MiniMax mode as the currently selected preset (FL2VA / R2VA / generic).
    current_mode = None
    for node in graph.get("nodes", []):
        for widget in node.get("widgets", []):
            if isinstance(widget, dict) and widget.get("name") == "preset_prompt":
                current_mode = _minimax_preset_mode(widget.get("value"))
    for match in re.finditer(r"(\d+)\s*(?:s|sec|secondi|seconds)\b", last_user, re.IGNORECASE):
        suffix = f"({match.group(1)}s)"
        for name in guides:
            if suffix not in name:
                continue
            mode = _minimax_preset_mode(name)
            # If we know the current mode, only add duration variants from the
            # same mode; otherwise keep only generic variants to avoid confusing
            # the LLM with multiple matching presets.
            if current_mode is None:
                if mode is None:
                    wanted.add(name)
            elif mode == current_mode:
                wanted.add(name)
    if not wanted:
        return ""
    # Suppress full-format video guides when the chat must only feed the enhancer.
    if has_images and _has_image_enhancer_target(graph):
        video_prefixes = ("MiniMax", "LTX", "Wan", "🎬 MiniMax", "🎞️ MiniMax", "🔄 MiniMax", "🎥 LTX", "🔀 LTX", "🎵 LTX", "📹 Wan", "🔄 Wan", "📖 Wan")
        wanted = {name for name in wanted if not any(name.startswith(prefix) for prefix in video_prefixes)}
        if not wanted:
            return ""
    parts = "\n\n".join(f"### {name}\n{guides[name]}" for name in sorted(wanted))
    return (
        "\n\nPROMPT WRITING GUIDES - when writing or editing a prompt for a node "
        "associated with one of these presets, follow the corresponding guide "
        "exactly, including its required output format. If the user asks for a "
        "different clip duration than the one the workflow is set to, set the "
        "node's \"duration\" widget (e.g. \"10s\") when it exists and "
        "update the workflow's duration/frame-count widgets accordingly:\n" + parts
    )


CHAT_GUIDES_PATH = Path(__file__).resolve().parent / "AILab_System_Prompts.json"


def _load_chat_guides():
    try:
        data = json.loads(CHAT_GUIDES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    guides = data.get("_chat_guides")
    return guides if isinstance(guides, dict) else {}


def _chat_guides_for(graph, has_images=False):
    """Inject workflow-knowledge guides whose trigger text appears in the
    snapshot's node types, titles, or widget values."""
    haystacks = []
    for node in graph.get("nodes", []):
        haystacks.append(str(node.get("type", "")))
        haystacks.append(str(node.get("title", "")))
        for widget in node.get("widgets", []):
            if isinstance(widget, dict) and isinstance(widget.get("value"), str):
                haystacks.append(widget["value"][:200])
    hay = "\n".join(haystacks).lower()
    parts = []
    for name, entry in _load_chat_guides().items():
        if not isinstance(entry, dict):
            continue
        triggers = entry.get("trigger") or []
        if isinstance(triggers, str):
            triggers = [triggers]
        text = entry.get("text")
        if isinstance(text, str) and any(str(t).lower() in hay for t in triggers):
            parts.append(f"### {name}\n{text}")
    for node in graph.get("nodes", []):
        widgets = {widget.get("name"): widget.get("value") for widget in node.get("widgets", []) if isinstance(widget, dict)}
        prompt_widget = next((name for name in ("prompt", "custom_prompt", "prompt_text") if name in widgets), None)
        if not prompt_widget or "passthrough" not in widgets:
            continue
        is_enhancer = has_images and "preset_prompt" in widgets and _is_image_enhancer_node(node)
        if is_enhancer:
            parts.append(
                f'### Exact image-enhancer target\nImage pixels are provided. Node {node.get("id")} exposes "{prompt_widget}", "preset_prompt", and "passthrough". '
                f'Inspect the provided image pixels to understand how the requested action applies. The node currently selects preset "{widgets.get("preset_prompt", "")}". '
                f'IGNORE any full prompt-writing guide for that preset above: the inner QwenVL node will use it to build the final prompt. '
                f'You MUST set node {node.get("id")} widget "{prompt_widget}" to a concise English action directive derived from the latest substantive request (skip execute-only confirmations; never copy it verbatim; never add the "For the target video..." binding line; never write integrated_multimodal_description/sections). '
                f'Do NOT change node {node.get("id")} widget "preset_prompt" unless the user explicitly asks to switch mode (e.g., "switch to I2VA"). '
                f'If the user only changes duration, keep the same preset family and update the duration widgets (the node\'s "duration" widget, plus value_1 or seconds on the sampler), not the preset_prompt. '
                f'set node {node.get("id")} widget "passthrough" to false, then queue. The inner QwenVL must analyze the image and create the final preset prompt.'
            )
        else:
            parts.append(
                f'### Exact passthrough target\nNode {node.get("id")} exposes both "{prompt_widget}" and "passthrough". '
                f'For generation, set node {node.get("id")} widget "{prompt_widget}" to the complete final English prompt, '
                f'set node {node.get("id")} widget "passthrough" to true, then queue. Do not omit either widget action.'
            )
    if not parts:
        return ""
    return (
        "\n\nWORKFLOW KNOWLEDGE - rules that apply to this workflow; follow them "
        "when choosing actions and values:\n" + "\n\n".join(parts)
    )


def _scan_model_files(folder, limit=500):
    try:
        import folder_paths
    except ImportError:
        return []
    found = []
    try:
        roots = folder_paths.get_folder_paths(folder) or []
    except Exception:
        return []
    for root in roots:
        base = Path(root)
        if not base.is_dir():
            continue
        try:
            for dirpath, _dirnames, filenames in os.walk(base):
                for name in filenames:
                    if name.lower().endswith((".safetensors", ".ckpt", ".pt", ".bin", ".gguf", ".onnx", ".pth")):
                        try:
                            found.append(str(Path(dirpath, name).relative_to(base).as_posix()))
                        except ValueError:
                            found.append(name)
                        if len(found) >= limit:
                            return sorted(found)
        except OSError:
            continue
    return sorted(found)


def installed_models_context():
    """Compact installed-models block for the prompt (cached). Best-effort:
    an unreadable store yields an empty block, never an error."""
    import time as _time
    now = _time.time()
    if now - _models_context_cache["at"] < MODELS_CONTEXT_TTL_S:
        return _models_context_cache["text"]
    parts = []
    for folder in MODELS_CONTEXT_FOLDERS:
        try:
            files = _scan_model_files(folder)
        except Exception:
            files = []
        shown = files[:MODELS_CONTEXT_FILES]
        extra = f" (+{len(files) - len(shown)} more)" if len(files) > len(shown) else ""
        parts.append(f"{folder} ({len(files)}{extra}): " + (", ".join(shown) if shown else "(none)"))
    text = "\n".join(parts)
    _models_context_cache.update(at=now, text=text)
    return text


def build_prompt(messages, graph, enable_thinking=False, has_images=False, has_video=False):
    history = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)
    snapshot = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
    if len(snapshot) > GRAPH_SNAPSHOT_CHARS:
        snapshot = snapshot[:GRAPH_SNAPSHOT_CHARS]
    models_block = installed_models_context()
    instruction = BASE_SYSTEM_PROMPT + _preset_guides(graph, messages, has_images) + _chat_guides_for(graph, has_images)
    instruction += THINKING_INSTRUCTION if enable_thinking else NO_THINKING_INSTRUCTION
    if has_video:
        instruction += "\nVIDEO INPUT: sampled frames from a video clip are attached to the latest user message. Treat them as the clip itself when the user asks to review, critique, or refine a generated video."
    if models_block:
        instruction += "\n\nINSTALLED MODELS — use these exact filenames in checkpoint/LoRA/VAE widgets (paths relative to their folders):\n" + models_block
    return f"{instruction}\n\nWORKFLOW SNAPSHOT:\n{snapshot}\n\nCONVERSATION:\n{history}\n\nJSON RESPONSE:"


_CONFIG_TRIGGER = re.compile(
    r"\b(?:use|usa|switch\s+to|passa\s+a|metti|set|con)\s+(?:the\s+|il\s+|la\s+)?"
    r"(native(?:\s+turbo)?|10\s*eros(?:[-\s]?max)?(?:\s+turbo)?|"
    r"singularity(?:\s+turbo)?|"
    r"r2va(?:\s+native)?(?:\s+turbo)?|"
    r"turbo(?:\s*lora)?(?:\s+(?:native|r2va(?:\s+native)?|10\s*eros(?:[-\s]?max)?))?|config\s*[abcd])\b",
    re.IGNORECASE,
)

_NATIVE_UNET = "minimax_h3_fl2va_pruned_nvfp4_convrot_int8.safetensors"
_R2VA_UNET = "minimax_h3_ref2va_pruned_nvfp4_convrot_int8.safetensors"
_SINGULARITY_UNET = "Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8.safetensors"
_EROS_UNET = "10Eros_Max_h3_hybrid_beta5_int8.safetensors"
# Needle "h3_hybrid" matches the non-turbo beta5 only; the fused file is
# named "h3_TURBO-hybrid_beta5" so it never collides.
_MINIMAX_CONFIGS = {
    "native": {
        "unet_needle": "fl2va_pruned",
        "unet_fallback": _NATIVE_UNET,
        "values": {"steps": 20, "sampler_name": "res_multistep", "scheduler": "simple", "shift_video": 12, "shift_audio": 3},
        "lora_mode": "bypass",
        "sparse_tau": 1.0,
    },
    "native_turbo": {
        "unet_needle": "fl2va_pruned",
        "unet_fallback": _NATIVE_UNET,
        "values": {"steps": 8, "sampler_name": "euler", "scheduler": "simple", "shift_video": 6, "shift_audio": 3},
        "lora_mode": "enable",
        "lora_needles": ["fl2v_turbo", "ref2v_turbo"],
        "sparse_tau": 1.3,
    },
    "10eros": {
        "unet_needle": "h3_hybrid_beta5",
        "unet_fallback": _EROS_UNET,
        "values": {"steps": 20, "sampler_name": "res_multistep", "scheduler": "simple", "shift_video": 12, "shift_audio": 3},
        "lora_mode": "bypass",
        "sparse_tau": 1.0,
    },
    "10eros_turbo": {
        "unet_needle": "h3_hybrid_beta5",
        "unet_fallback": _EROS_UNET,
        "values": {"steps": 8, "sampler_name": "euler", "scheduler": "simple", "shift_video": 6, "shift_audio": 3},
        "lora_mode": "enable",
        "lora_needles": ["fusion_turbo"],
        "sparse_tau": 1.3,
    },
    # R2VA variants: same sampler configs, but the ref2va unet (image-ref
    # conditioned) and the ref2v turbo LoRA for the turbo step.
    "r2va_native": {
        "unet_needle": "ref2va_pruned",
        "unet_fallback": _R2VA_UNET,
        "preset_mode": "R2VA",
        "values": {"steps": 20, "sampler_name": "res_multistep", "scheduler": "simple", "shift_video": 12, "shift_audio": 3},
        "lora_mode": "bypass",
        "sparse_tau": 1.0,
    },
    "r2va_native_turbo": {
        "unet_needle": "ref2va_pruned",
        "unet_fallback": _R2VA_UNET,
        "preset_mode": "R2VA",
        "values": {"steps": 8, "sampler_name": "euler", "scheduler": "simple", "shift_video": 6, "shift_audio": 3},
        "lora_mode": "enable",
        "lora_needles": ["ref2v_turbo", "fl2v_turbo"],
        "sparse_tau": 1.3,
    },
    # Singularity fused ref2va: ref-conditioning is baked in — no rank-256 LoRA.
    # Sigma node stays bypassed in the wf (native schedule wins on this DiT);
    # turbo = ref2v 4-step LoRA at 6 steps euler/beta.
    "r2va_singularity": {
        "unet_needle": "singularity",
        "unet_fallback": _SINGULARITY_UNET,
        "preset_mode": "R2VA",
        "values": {"steps": 20, "sampler_name": "res_multistep", "scheduler": "simple"},
        "lora_mode": "bypass",
        "sigma_mode": "bypass",
        "sparse_tau": 1.0,
    },
    "r2va_singularity_turbo": {
        "unet_needle": "singularity",
        "unet_fallback": _SINGULARITY_UNET,
        "preset_mode": "R2VA",
        "values": {"steps": 6, "sampler_name": "euler", "scheduler": "beta"},
        "lora_mode": "enable",
        "lora_needles": ["ref2v_turbo_4step"],
        "sigma_mode": "bypass",
        "sparse_tau": 1.3,
    },
}


def _last_user_message(messages):
    for item in reversed(messages or []):
        if item.get("role") == "user" and isinstance(item.get("content"), str) and item["content"].strip():
            return item["content"].strip()
    return ""


def _widget_options(widget):
    return (widget.get("options") or {}).get("values") or []


def _match_option(widget, needle):
    for option in _widget_options(widget):
        if needle in str(option).lower():
            return option
    return None


def _minimax_preset_mode(preset_name):
    preset_name = str(preset_name or "").lower()
    if "fl2va" in preset_name:
        return "FL2VA"
    if "r2va" in preset_name:
        return "R2VA"
    return None


def _match_minimax_preset(widget, seconds, current_preset):
    """Pick a duration variant that stays in the same MiniMax mode as the
    currently selected preset (FL2VA/R2VA/generic). Falls back to the generic
    duration variant when the current preset is generic."""
    options = _widget_options(widget)
    suffix = f"({seconds}s)"
    mode = _minimax_preset_mode(current_preset)
    if mode:
        for option in options:
            if suffix in str(option) and mode.lower() in str(option).lower():
                return option
        # If no same-mode variant exists, don't fall back to a different mode.
        return None
    # Current preset is generic (T2VA/I2VA): prefer a generic variant without
    # an FL2VA/R2VA marker.
    for option in options:
        if suffix in str(option) and _minimax_preset_mode(option) is None:
            return option
    # Last resort: any duration match.
    for option in options:
        if suffix in str(option):
            return option
    return None


def _minimax_result(graph, config_key, text):
    """Apply a MiniMax H3 sampler config + optional cleaned scene directive +
    queue on the enhancer node. `text` is the raw user message (duration is
    parsed from it, the cleaned remainder becomes the prompt directive)."""
    for node in graph.get("nodes", []):
        widgets = {w.get("name"): w for w in node.get("widgets", []) if isinstance(w, dict)}
        if not {"unet_name", "preset_prompt", "passthrough"}.issubset(widgets):
            continue
        config = _MINIMAX_CONFIGS[config_key]
        actions = []
        unet_widget = widgets["unet_name"]
        if config["unet_needle"]:
            unet = _match_option(unet_widget, config["unet_needle"]) or config["unet_fallback"]
            actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "unet_name", "value": unet})
        target_mode = config.get("preset_mode")
        effective_preset = str(widgets["preset_prompt"].get("value", ""))
        if target_mode:
            # R2VA configs must also switch the enhancer preset to an R2VA
            # variant — an FL2VA-style prompt would carry the wrong bindings
            # for the ref-conditioned unet. Preserve the "(Ns)" duration.
            if _minimax_preset_mode(effective_preset) != target_mode:
                dur_match = re.search(r"\((\d+s)\)\s*$", effective_preset)
                dur_suffix = dur_match.group(1) if dur_match else None
                for opt in _widget_options(widgets["preset_prompt"]):
                    opt_str = str(opt)
                    if _minimax_preset_mode(opt_str) == target_mode and (dur_suffix is None or f"({dur_suffix})" in opt_str):
                        actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "preset_prompt", "value": opt_str})
                        effective_preset = opt_str
                        break
        for name, value in config["values"].items():
            widget = widgets.get(name)
            if widget is None:
                continue
            options = _widget_options(widget)
            if options and str(value) not in [str(o) for o in options]:
                match = _match_option(widget, str(value).lower())
                if match is None:
                    continue
                value = match
            actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": name, "value": value})
        seconds = _clip_duration_seconds(text)
        if seconds is not None:
            if "value_1" in widgets:
                actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "value_1", "value": seconds})
            if "duration" in widgets:
                # New-style node: duration is its own widget; the preset stays
                # on the same family name regardless of clip length.
                dur_value = f"{seconds}s"
                dur_options = [str(o) for o in _widget_options(widgets["duration"])]
                if not dur_options or dur_value in dur_options:
                    actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "duration", "value": dur_value})
            else:
                # Match duration inside the same mode family of the preset the
                # config just selected (effective_preset), not the stale widget
                # value — otherwise an FL2VA current preset would flip an R2VA
                # config back to FL2VA.
                preset_match = _match_minimax_preset(widgets["preset_prompt"], seconds, effective_preset)
                if preset_match:
                    actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "preset_prompt", "value": preset_match})
        directive = _clean_action_directive(text)
        if directive and "prompt" in widgets:
            actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "prompt", "value": directive})
        lora_mode = config.get("lora_mode")
        prefix = f'{node["id"]}:'
        # Style LoRA slots (realism / char-swap on Singularity) are user
        # choices — config switches must not rename or bypass them.
        def _style_lora(inner):
            for w in inner.get("widgets", []):
                if isinstance(w, dict) and w.get("name") == "lora_name":
                    v = str(w.get("value") or "").lower()
                    return "realism" in v or "character_swap" in v
            return False
        if lora_mode:
            target_mode = 0 if lora_mode == "enable" else 4
            for inner in graph.get("nodes", []):
                inner_id = str(inner.get("id", ""))
                if not inner_id.startswith(prefix) or "lora" not in str(inner.get("type", "")).lower():
                    continue
                if _style_lora(inner):
                    continue
                if inner.get("mode", 0) != target_mode:
                    actions.append({"type": "set_node_mode", "node_id": inner["id"], "mode": lora_mode})
        # Turbo configs also pick the right LoRA file: the lightx2v per-mode
        # LoRA for Native (matched to the FL2VA/R2VA preset family), the
        # TenStrip combined fusion LoRA for 10Eros.
        lora_needles = config.get("lora_needles")
        if lora_needles:
            needles = list(lora_needles)
            preset_mode = _minimax_preset_mode(widgets["preset_prompt"].get("value"))
            if preset_mode == "R2VA":
                needles.sort(key=lambda n: "ref2v" not in n)
            elif preset_mode == "FL2VA":
                needles.sort(key=lambda n: "fl2v" not in n)
            for inner in graph.get("nodes", []):
                inner_id = str(inner.get("id", ""))
                if not inner_id.startswith(prefix) or "lora" not in str(inner.get("type", "")).lower():
                    continue
                if _style_lora(inner):
                    continue
                for w in inner.get("widgets", []):
                    if not isinstance(w, dict) or w.get("name") != "lora_name":
                        continue
                    for needle in needles:
                        match = _match_option(w, needle)
                        if match:
                            if w.get("value") != match:
                                actions.append({"type": "set_widget_value", "node_id": inner["id"], "widget": "lora_name", "value": match})
                            break
        sigma_mode = config.get("sigma_mode")
        if sigma_mode:
            target_mode = 0 if sigma_mode == "enable" else 4
            for inner in graph.get("nodes", []):
                inner_id = str(inner.get("id", ""))
                if not inner_id.startswith(prefix) or "sigmashift" not in str(inner.get("type", "")).lower().replace("_", ""):
                    continue
                if inner.get("mode", 0) != target_mode:
                    actions.append({"type": "set_node_mode", "node_id": inner["id"], "mode": sigma_mode})
        sparse_tau = config.get("sparse_tau")
        if sparse_tau is not None:
            for inner in graph.get("nodes", []):
                inner_id = str(inner.get("id", ""))
                if not inner_id.startswith(prefix) or "blocksparse" not in str(inner.get("type", "")).lower().replace("_", ""):
                    continue
                inner_widgets = {w.get("name") for w in inner.get("widgets", []) if isinstance(w, dict)}
                for wname in ("selection.tau", "tau"):
                    if wname in inner_widgets:
                        actions.append({"type": "set_widget_value", "node_id": inner["id"], "widget": wname, "value": sparse_tau})
                        break
        actions.append({"type": "set_widget_value", "node_id": node["id"], "widget": "passthrough", "value": False})
        actions.append({"type": "queue_workflow"})
        label = {"native": "Native", "native_turbo": "Native Turbo",
                 "10eros": "10Eros", "10eros_turbo": "10Eros Turbo",
                 "r2va_native": "R2VA Native", "r2va_native_turbo": "R2VA Native Turbo",
                 "r2va_singularity": "R2VA Singularity", "r2va_singularity_turbo": "R2VA Singularity Turbo"}[config_key]
        if re.match(r"^\s*(usa|passa|metti|fai|genera|crea)\b", text, re.IGNORECASE):
            message = f"⚙️ MiniMax H3 → {label}. Workflow in coda."
        else:
            message = f"⚙️ MiniMax H3 → {label}. Workflow queued."
        return {"message": message, "actions": actions, "choices": []}
    return None


def _fix_minimax_preset_actions(result, graph):
    """After LLM routing, correct preset_prompt actions on MiniMax H3 nodes so
    the duration variant matches the currently selected mode (FL2VA/R2VA).
    This prevents the model from accidentally switching a FL2VA workflow to the
    generic I2VA preset when the user only mentions a duration."""
    nodes_by_id = {str(node.get("id")): node for node in graph.get("nodes", [])}
    for action in result.get("actions", []):
        if action.get("type") != "set_widget_value" or action.get("widget") != "preset_prompt":
            continue
        node_id = str(action.get("node_id"))
        node = nodes_by_id.get(node_id)
        if not node:
            continue
        widgets = {w.get("name"): w for w in node.get("widgets", []) if isinstance(w, dict)}
        if not {"unet_name", "preset_prompt", "passthrough"}.issubset(widgets):
            continue
        current_preset = str(widgets["preset_prompt"].get("value") or "")
        current_mode = _minimax_preset_mode(current_preset)
        if not current_mode:
            # Current preset is the generic one; don't second-guess explicit mode switches.
            continue
        new_value = str(action.get("value") or "")
        # Normalize legacy dropdown names to the canonical preset so mode
        # comparison and widget options work with the new naming.
        resolved_new, resolved_dur = resolve_preset(new_value, VL_ALIASES)
        new_mode = _minimax_preset_mode(resolved_new)
        if resolved_new in [str(o) for o in _widget_options(widgets["preset_prompt"])]:
            action["value"] = resolved_new
        def _emit_duration(dur):
            if not dur or "duration" not in widgets:
                return
            dur_options = [str(o) for o in _widget_options(widgets["duration"])]
            if not dur_options or dur in dur_options:
                result["actions"].append({"type": "set_widget_value", "node_id": action.get("node_id"), "widget": "duration", "value": dur})

        if new_mode == current_mode:
            # Same family — duration is expressed via the "duration" widget.
            _emit_duration(resolved_dur)
            continue
        if "duration" in widgets:
            # New-style node: keep the current preset family, duration lives in
            # its own widget.
            action["value"] = current_preset
            _emit_duration(resolved_dur)
            continue
        duration_match = re.search(r"\((\d+)s\)", new_value)
        if not duration_match:
            # Legacy node without a duration suffix; revert to the current
            # preset to stay safe.
            action["value"] = current_preset
            continue
        fixed = _match_minimax_preset(widgets["preset_prompt"], int(duration_match.group(1)), current_preset)
        if fixed:
            action["value"] = fixed
    return result


def _explicit_minimax_request(messages, graph):
    """Deterministic MiniMax config switch ("use native/10Eros/turbo", "config A/B/C"):
    applies sampler widgets + duration + cleaned action locally, no LLM call."""
    last_user = _last_user_message(messages)
    trigger = _CONFIG_TRIGGER.search(last_user)
    if not trigger:
        return None
    raw = re.sub(r"\s+", " ", trigger.group(1).lower()).strip()
    if "config" in raw:
        config_key = {"config a": "10eros_turbo", "config b": "native_turbo",
                      "config c": "native", "config d": "10eros"}.get(raw)
        if config_key is None:
            return None
    elif "eros" in raw:
        config_key = "10eros_turbo" if "turbo" in raw else "10eros"
    elif "singularity" in raw or "r2va" in raw:
        # R2VA in production = Singularity fused checkpoint; the legacy ref2va
        # + rank-256 configs stay selectable via directives only.
        config_key = "r2va_singularity_turbo" if "turbo" in raw else "r2va_singularity"
    elif "native" in raw:
        config_key = "native_turbo" if "turbo" in raw else "native"
    elif "turbo" in raw:
        # Bare "turbo" keeps the legacy meaning: native model + turbo LoRA.
        config_key = "native_turbo"
    else:
        return None
    return _minimax_result(graph, config_key, last_user)


def _comfy_queue_busy():
    """True while ComfyUI has a job running or pending. Fail-open (False) —
    if the snapshot can't be read, don't block chat on it."""
    try:
        with urllib.request.urlopen(COMFY_URL.rstrip("/") + "/queue", timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        print(f"[QwenVL Chat] queue snapshot failed ({error}); proceeding")
        return False
    return bool(data.get("queue_running") or data.get("queue_pending"))


def _comfy_vram_free():
    """ComfyUI's /system_stats device 0 (vram_free, vram_total) in bytes, or
    None when that endpoint is unavailable (older servers, IPC hiccup)."""
    try:
        with urllib.request.urlopen(COMFY_URL.rstrip("/") + "/system_stats", timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    devices = (data or {}).get("devices") or []
    if not devices:
        return None
    device = devices[0] or {}
    free = device.get("vram_free")
    total = device.get("vram_total")
    if isinstance(free, (int, float)) and isinstance(total, (int, float)) and total > 0:
        return float(free), float(total)
    return None


def _llamaswap_running():
    """Set of model names llama-swap currently holds (any state), or None
    when it can't be reached or answers an unexpected shape (fail open)."""
    try:
        with urllib.request.urlopen(LLAMA_SWAP_URL.rstrip("/") + "/running", timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    models = data.get("running") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return None
    names = set()
    for entry in models:
        if isinstance(entry, dict) and isinstance(entry.get("model"), str):
            names.add(entry["model"])
    return names


def _comfy_free(model_name=None):
    """Make room on the shared card for the api chat model.

    Three states, in order:
    1. The requested model is already up in llama-swap: serve from it — no
       load happens, so resident VRAM (a warm NInfer alone holds ~22.6 of
       23.6 GiB) must not block the message.
    2. Otherwise unload ComfyUI via POST /free {'unload_models','free_memory'}
       — the route 500s on an empty body, which made the original version a
       silent no-op — and poll /system_stats until the device frees: the
       queue worker applies the flags asynchronously. Fail-open when stats
       are unreadable.
    3. Still occupied after the wait: other llama-swap models are fine (the
       broker swaps them out for our request), but with llama-swap empty the
       memory belongs to something it cannot reclaim (ComfyUI failed to
       release, or another process) — stop with an actionable error instead
       of letting the llama-server die on the load ("upstream command exited
       prematurely")."""
    running = _llamaswap_running()
    if running and model_name and model_name in running:
        return
    payload = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
    try:
        request = urllib.request.Request(
            COMFY_URL.rstrip("/") + "/free", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except Exception as error:
        print(f"[QwenVL Chat] ComfyUI /free failed ({error})")
        return
    stats = _comfy_vram_free()
    if stats is None:
        return
    free, total = stats
    required = total * FREE_VRAM_MIN_FRACTION
    deadline = time.time() + FREE_WAIT_S
    while free < required and time.time() < deadline:
        time.sleep(FREE_POLL_S)
        stats = _comfy_vram_free()
        if stats is None:
            return
        free, total = stats
        required = total * FREE_VRAM_MIN_FRACTION
    if free < required:
        running = _llamaswap_running()
        if running is None or running:
            return
        raise ValueError(
            "ComfyUI (or another local process) is still holding most of the GPU "
            f"({round(free / 1073741824, 1)} GB free of {round(total / 1073741824, 1)} GB; "
            f"the chat model needs about {round(required / 1073741824, 1)} GB). "
            "Restart ComfyUI (or use its Free model and node cache menu) and send again."
        )


def _await_comfy_idle(model_name=None):
    if not WAIT_FOR_IDLE:
        return
    while _comfy_queue_busy():
        time.sleep(QUEUE_POLL_S)
    _comfy_free(model_name)


def fit_history_to_budget(messages):
    """Drop oldest messages until the history fits PROMPT_BUDGET_CHARS.
    The latest message (the actual request) is never dropped."""
    total = sum(len(item["content"]) for item in messages)
    index = 0
    while total > PROMPT_BUDGET_CHARS and len(messages) - index > 1:
        total -= len(messages[index]["content"])
        index += 1
    return messages[index:]


def _api_models():
    """Model list for the API backend. Static default so the dropdown works
    even when llama-swap is down; the configured model is always the default."""
    return [API_DEFAULT_MODEL]


CIVITAI_API = "https://api.civitai.com/v1/models"


def civitai_search(query, types="", limit=8):
    """Proxy CivitAI model search (fixed host, capped). Returns compact hits
    with per-version download URLs the download action accepts."""
    try:
        count = max(1, min(int(limit), 20))
    except (TypeError, ValueError):
        count = 8
    params = {"query": query, "limit": count, "nsfw": "true", "sort": "Highest Rated"}
    if types:
        params["types"] = types
    url = CIVITAI_API + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "QwenChat/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))
    items = []
    for model in (data or {}).get("items") or []:
        versions = []
        for version in (model.get("modelVersions") or [])[:3]:
            files = [f for f in (version.get("files") or []) if f.get("downloadUrl")]
            if not files:
                continue
            first = files[0]
            versions.append({
                "id": version.get("id"),
                "name": version.get("name"),
                "url": first.get("downloadUrl"),
                "sizeKB": first.get("sizeKB"),
                "baseModel": version.get("baseModel"),
            })
        if versions:
            items.append({"name": model.get("name"), "type": model.get("type"), "versions": versions})
    return items


def _content_filename(headers):
    raw = headers.get("Content-Disposition", "")
    match = re.search(r"filename\*\s*=\s*UTF-8''([^;]+)", raw, re.IGNORECASE)
    if not match:
        match = re.search(r'filename\s*=\s*"?([^";]+)"?', raw, re.IGNORECASE)
    if not match:
        return None
    try:
        return urllib.parse.unquote(match.group(1).strip())
    except Exception:
        return None


def download_model(url, kind, filename=None, token=None):
    """Fetch a model file into the ComfyUI store. Guardrails: initial host
    allow-list, kind-to-folder allow-map, extension allow-list per kind,
    sanitized basename, auto-suffix on collision. Token (if any) travels only
    as an Authorization header and is never logged."""
    if kind not in DOWNLOAD_KINDS:
        raise ValueError(f"unknown model kind '{kind}'")
    folder, exts = KIND_FOLDERS[kind]
    try:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
    except Exception:
        raise ValueError("unparseable download URL")
    if host not in DOWNLOAD_HOSTS:
        raise ValueError(f"downloads only allowed from {sorted(DOWNLOAD_HOSTS)}")
    try:
        import folder_paths as _fp
        roots = _fp.get_folder_paths(folder) or []
    except Exception:
        roots = []
    if not roots:
        raise ValueError(f"no local folder for model kind '{kind}'")
    dest_dir = roots[0]
    headers = {"User-Agent": "QwenChat/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=120)
    except Exception as error:
        raise ValueError(f"download request failed: {error}")
    with response:
        name = filename or _content_filename(response.headers) or os.path.basename(urllib.parse.urlsplit(url).path)
        base = (name or "").strip()
        if not base or base in {".", ".."} or "/" in base or "\\" in base:
            raise ValueError("could not determine a safe filename (pass one explicitly)")
        if not base.lower().endswith(exts):
            raise ValueError(f"'{base}' is not a {kind} file (expected {exts})")
        target = os.path.join(dest_dir, base)
        stem, dot, ext = base.rpartition(".")
        counter = 2
        while os.path.exists(target):
            target = os.path.join(dest_dir, f"{stem}-{counter}{dot}{ext}")
            counter += 1
        total = 0
        with open(target, "wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
    return {"filename": os.path.basename(target), "bytes": total, "kind": kind, "folder": folder}


def _api_chat_completions(system, user_text, images_b64, model_name, options):
    """POST one OpenAI-compatible chat request to llama-swap. Images go as
    data-URL parts on the user message (NInfer-HauHauCS-27B takes vision)."""
    content = [{"type": "text", "text": user_text}]
    for b64 in images_b64 or []:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "max_tokens": int(options.get("max_tokens", 1024)),
        "temperature": float(options.get("temperature", 0.2)),
        "top_p": float(options.get("top_p", 0.9)),
    }
    try:
        seed = int(options.get("seed", 1))
    except (TypeError, ValueError):
        seed = None
    if seed is not None:
        payload["seed"] = seed
    body = json.dumps(payload).encode("utf-8")
    url = API_BASE_URL.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_S) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8")[:500]
        except Exception:
            detail = ""
        raise ValueError(f"llama-swap API error HTTP {error.code}: {detail or error.reason}")
    except urllib.error.URLError as error:
        raise ValueError(f"llama-swap not reachable at {API_BASE_URL} ({error.reason}). Is llama-swap running?")
    except TimeoutError:
        raise ValueError(f"llama-swap request timed out after {API_TIMEOUT_S}s (cold model load can take ~1 min; retry once).")
    choices = (data or {}).get("choices") or []
    message = (choices[0].get("message") if choices else None) or {}
    text = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    if not isinstance(text, str):
        text = str(text)
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)
    if not text.strip() and reasoning.strip():
        text = reasoning
    elif reasoning.strip() and text.strip() and reasoning.strip() not in text:
        # NInfer (preserve-thinking) reasons in a separate field: fold it
        # into the <think> block so extract_thinking surfaces it in the
        # panel's Thinking view instead of dropping it.
        text = f"{THINK_OPEN}{reasoning.strip()}{THINK_CLOSE}\n{text}"
    if not text.strip():
        raise ValueError(f"llama-swap returned an empty reply: {str(data)[:500]}")
    return text


class ChatRuntime:
    def __init__(self):
        self._instances = {}
        self._lock = threading.Lock()

    def models(self):
        hf = sys.modules.get("AILab_QwenVL")
        gguf = sys.modules.get("AILab_QwenVL_GGUF")
        # HF_ALL_MODELS includes both VL and text-only models (Qwen3.5/3.8).
        hf_models = sorted((getattr(hf, "HF_ALL_MODELS", {}) or {}).keys()) if hf else []
        gguf_models = sorted(((getattr(gguf, "GGUF_VL_CATALOG", {}) or {}).get("models") or {}).keys()) if gguf else []
        return {"hf": hf_models, "gguf": gguf_models, "api": _api_models()}

    def chat(self, backend, model_name, messages, graph, options, images=None, video=None, directives=None):
        messages = validate_messages(messages)
        messages = fit_history_to_budget(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                expanded = _expand_wildcard_tokens(messages[i].get("content", ""))
                if expanded != messages[i].get("content"):
                    messages[i] = {**messages[i], "content": expanded}
                break
        graph = validate_graph(graph)
        images = validate_images(images or [])
        video = validate_images(video or [], MAX_VIDEO_FRAMES)
        directives = directives if isinstance(directives, dict) else {}
        sel_config = str(directives.get("config") or "auto")
        sel_text = str(directives.get("text") or "")
        explicit = None
        if sel_config in _MINIMAX_CONFIGS:
            explicit = _minimax_result(graph, sel_config, sel_text)
        if explicit is None:
            explicit = _explicit_minimax_request(messages, graph)
        if explicit is not None:
            return explicit
        available = self.models().get(backend)
        if available is None:
            raise ValueError("backend must be hf, gguf, or api")
        if not available:
            raise ValueError(f"no models available for backend '{backend}'")
        if not model_name:
            model_name = available[0]
        elif model_name not in available:
            raise ValueError(f"unknown model '{model_name}'")
        enable_thinking = bool(options.get("thinking", False))
        options = dict(options)
        if enable_thinking:
            # Thinking consumes tokens before the JSON reply; a small budget
            # truncates the response after the reasoning and no JSON is emitted.
            options["max_tokens"] = max(int(options.get("max_tokens", 1024)), 4096)
        prompt = build_prompt(messages, graph, enable_thinking, bool(images), bool(video))
        with self._lock:
            text = self._generate(backend, model_name, prompt, options, enable_thinking, images, video)
        MAX_JSON_ATTEMPTS = 3
        result = parse_model_response(text)
        attempts = 1
        while not result.get("parsed") and attempts < MAX_JSON_ATTEMPTS:
            attempts += 1
            if attempts == 2:
                reminder = (
                    "FAILURE: You did not return a JSON object. Output EXACTLY one JSON object and nothing else:\n"
                    '{"message":"Brief summary of fixes","actions":[{"type":"set_widget_value","node_id":12,"widget":"steps","value":20}],"choices":[]}\n'
                    "No prose, no analysis, no think tags, no markdown fences. Copy this structure, fill in the actual actions, output only the JSON."
                )
            else:
                reminder = (
                    "FINAL FAILURE: Your previous replies were not JSON objects. Output exactly this structure and nothing else:\n"
                    '{"message":"Fixed KSampler widgets","actions":[{"type":"set_widget_value","node_id":12,"widget":"control_after_generate","value":"fixed"},{"type":"set_widget_value","node_id":12,"widget":"steps","value":20},{"type":"set_widget_value","node_id":12,"widget":"cfg","value":1},{"type":"set_widget_value","node_id":12,"widget":"sampler_name","value":"euler"},{"type":"set_widget_value","node_id":12,"widget":"scheduler","value":"simple"}],"choices":[]}\n'
                    "Replace the example node ids and values with the actual fixes. No prose before or after. Only JSON."
                )
            retry_messages = messages + [
                {"role": "assistant", "content": (text or "")[:2000]},
                {"role": "user", "content": "Your reply was not a JSON object and applied no workflow changes. " + reminder},
            ]
            retry_prompt = build_prompt(retry_messages, graph, enable_thinking, bool(images), bool(video))
            with self._lock:
                retry_text = self._generate(backend, model_name, retry_prompt, options, enable_thinking, images, video)
            text = retry_text
            result = parse_model_response(text)
        # Keep "parsed" in the payload so the frontend can tell whether the
        # model actually returned workflow actions (and retry if it didn't).
        result = enforce_image_enhancer_routing(result, graph, messages, bool(images))
        result = _fix_minimax_preset_actions(result, graph)
        return enforce_image_reference_bindings(result, graph, bool(images))

    def _generate(self, backend, model_name, prompt, options, enable_thinking, images, video=None):
        if backend == "hf":
            return self._chat_hf(model_name, prompt, options, enable_thinking, images, video)
        if backend == "api":
            return self._chat_api(model_name, prompt, options, enable_thinking, images, video)
        return self._chat_gguf(model_name, prompt, options, enable_thinking, images, video)

    def _chat_api(self, model_name, prompt, options, enable_thinking=False, images=None, video=None):
        # Ping-pong first: a finished workflow leaves diffusion weights
        # resident, and a running one owns the card — wait behind the job,
        # free the weights, then load the chat model into a free card.
        # (Runs under ChatRuntime's lock; the retry path re-checks cheaply.)
        _await_comfy_idle(model_name)
        # Reuse the exact prompt the local backends get (system rules +
        # workflow snapshot + conversation) as the user message, with the
        # protocol system prompt as `system` — so workflow control (actions,
        # choices, bindings) behaves identically regardless of backend.
        system = BASE_SYSTEM_PROMPT + (THINKING_INSTRUCTION if enable_thinking else NO_THINKING_INSTRUCTION)
        images_b64 = [base64.b64encode(data).decode("ascii") for data in (images or []) + (video or [])]
        return _api_chat_completions(system, prompt, images_b64, model_name, options)

    def _chat_hf(self, model_name, prompt, options, enable_thinking=False, images=None, video=None):
        module = sys.modules["AILab_QwenVL"]
        instance = self._instances.get("hf")
        if instance is None:
            instance = module.QwenVLBase()
            self._instances["hf"] = instance
        instance.load_model(
            model_name,
            options.get("quantization", module.Quantization.Q8.value),
            options.get("attention_mode", "auto"),
            False,
            options.get("device", "auto"),
            True,
        )
        pil_images = _decode_images(images or [])
        image = pil_images[0] if len(pil_images) > 0 else None
        image2 = pil_images[1] if len(pil_images) > 1 else None
        image3 = pil_images[2] if len(pil_images) > 2 else None
        pil_frames = _decode_images(video or [])
        return instance.generate(
            prompt, image, image2, image3, len(pil_frames) or 1,
            int(options.get("max_tokens", 1024)),
            float(options.get("temperature", 0.2)),
            float(options.get("top_p", 0.9)),
            1,
            float(options.get("repetition_penalty", 1.05)),
            model_name=model_name,
            video=pil_frames or None,
            enable_thinking=enable_thinking,
        )

    def _chat_gguf(self, model_name, prompt, options, enable_thinking=False, images=None, video=None):
        module = sys.modules["AILab_QwenVL_GGUF"]
        instance = self._instances.get("gguf")
        if instance is None:
            instance = module.QwenVLGGUFBase()
            self._instances["gguf"] = instance
        instance._load_model(
            model_name,
            options.get("device", "auto"),
            options.get("ctx"),
            options.get("n_batch"),
            options.get("gpu_layers"),
            options.get("image_max_tokens"),
            options.get("top_k"),
            options.get("pool_size"),
        )
        # `images`/`video` arrive already validated to bytes by chat()
        images_b64 = [base64.b64encode(data).decode("ascii") for data in (images or []) + (video or [])]
        return instance._invoke(
            SYSTEM_PROMPT,
            prompt,
            images_b64,
            int(options.get("max_tokens", 1024)),
            float(options.get("temperature", 0.2)),
            float(options.get("top_p", 0.9)),
            float(options.get("repetition_penalty", 1.05)),
            int(options.get("seed", 1)),
            model_name,
            enable_thinking=enable_thinking,
        )

    def unload(self, backend="all"):
        if backend not in {"hf", "gguf", "api", "all"}:
            raise ValueError("backend must be hf, gguf, api, or all")
        with self._lock:
            targets = list(self._instances) if backend == "all" else [backend]
            for target in targets:
                instance = self._instances.pop(target, None)
                if instance is not None:
                    instance.clear()
            gc.collect()
        return targets


CHAT_RUNTIME = ChatRuntime()
