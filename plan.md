# Plan: QwenChat as a native ComfyUI agent

Branch `feat/native-agent`, based on `review/qwenchat-reliability` (which carries
the new `tests/test_chat_reliability.py` plus the runbook notes in
`~/wt/orca-specs/out/qwenchat.md`). Work ships in three slices, one commit each,
then a draft PR against `T-Crypt/ComfyUI-QwenVL-Mod`'s default branch.

## What exists today

`chat_service.py` runs one request/response protocol. The panel posts the
conversation plus a live graph snapshot to `POST /qwenvl/chat`. The model answers
with one JSON object that carries `message`, `actions` and `choices`.
`validate_actions` is a hand-rolled gate over `ALLOWED_ACTIONS`. The panel's
`applyActions` executes those actions against `app.graph` (add, remove, connect,
disconnect, set widget value, layout, queue, model search and download, the
slice-3 queue and workflow-library actions).

The chat model is `NInfer-HauHauCS-27B` on llama-swap at
`http://127.0.0.1:9090/v1`. It advertises `function_calling: true`, `vision:
true` and `supported_parameters: [tools, tool_choice]`. It is served with
`--tolerant-tool-calls`.

The review's finding 2 is the top gap to close first: `_api_chat_completions`
makes one `urlopen` attempt, so a chat that races a cold model load eats a hard
`HTTP 502/503` failure.

## Gaps against the required capabilities

1. Live graph read exists (`snapshotGraph` over `app.graph._nodes`). No API-format
   read.
2. Live graph edits exist but are not undoable as one step and are not
   highlighted.
3. No installed-node context from `GET /object_info`, and no suggest plus
   preview-diff flow.
4. Queue exists client-side, but no ComfyUI websocket progress and no result
   images in the chat.
5. Save, load and delete exist. Rename does not.
6. `/qwenvl/chat/models` returns only the one default API model. The other
   llama-swap models are missing from the selector.

## Data shape

The single source of truth is a declarative tool registry.

```
Tool = {
  name:      str,      # the canonical action type, e.g. "connect"
  summary:   str,      # one line, used to build the system prompt
  schema:    dict,     # JSON Schema (draft-07 subset) for the arguments
  normalize: fn(raw) -> canonical action | None,
  frontend:  bool,     # True: the panel runs it; False: server-only
}
```

Derived from `TOOLS`:

- `TOOL_SCHEMAS`, an OpenAI `tools` array (`[{type: function, function: {...}}]`).
- the action section of the system prompt.
- `validate_actions`, which now walks the registry instead of a long `if` chain.

The canonical action stays the same dict the panel already consumes. Tool calls
are one transport, the panel sees one channel. The server translates an OpenAI
`tool_calls` reply into the same `actions` array, so the durability review's
"pick one owner" note holds.

A `suggest_workflow` tool carries a structured plan so the panel can render a
preview diff before anything touches the canvas.

```
{ "type": "suggest_workflow", "summary": str, "title": str,
  "steps": [ {"key": str, "node_type": str, "widgets": {name: value} } ],
  "links": [ {"from": key|id, "out": str, "to": key|id, "in": str} ] }
```

A server session checkpoint keeps the conversation durable across a GPU handoff.

```
_SESSIONS[session_id] = { "at": float, "messages": [...], "model": str }
```

Capped at 32 sessions, 8 hour TTL.

## Slices

### Slice 1, reliability and the cheap test fix

- Retry `HTTP 502/503` and connection-refused `URLError` inside
  `_api_chat_completions` only, twice with a 3 second base backoff and a hard cap.
  Server owns transport retries, so the panel's three-round loop never resends
  one.
- Fix `test_hf_chat_forwards_video_frames`, which reads `args[3]` (image3) rather
  than `args[4]` (frame_count) from the fake `generate`.
- Tests: new retry cases in `tests/test_chat_reliability.py`. Suite green apart
  from nothing.

### Slice 2, tool registry and native tool calls

- `TOOLS` registry, `TOOL_SCHEMAS`, prompt generation, registry-driven
  `validate_actions`.
- Pass `tools` to llama-swap and accept `tool_calls`, folded back into `actions`.
- `GET /qwenvl/chat/tools` serves the schemas.
- llama-swap model list in `_api_models`, default first, static fallback.
- Tests: registry schema validity, tool-call translation, models list with a
  mocked llama-swap.

### Slice 3, live graph, queue progress and library rename

- Frontend: group each agent turn as one undo step
  (`app.canvas.emitBeforeChange` / `emitAfterChange`) and highlight touched nodes.
- Frontend: queue through `app.graphToPrompt()` plus `POST /prompt`, follow the
  ComfyUI websocket for progress, and render result images in the chat.
- Frontend: render the `suggest_workflow` preview diff with a Build button.
- Server: `workflow_rename` action and tool, `node_types_context()` from
  `GET /object_info`, `GET /qwenvl/chat/status` for the handoff banner,
  conversation checkpoint routes.
- Tests: rename validation, node context never raises, status route shape.

## Verification

Offline only. `python3 -m unittest discover -s tests` covers the protocol,
reliability and the new server routes, with every HTTP call mocked. No prompt is
queued and no model is loaded from this work. The operator runs the live
acceptance steps at the end, and ComfyUI is restarted only through its berth unit
and only when `GET /queue` is empty.
