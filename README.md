# Unity QA Agent

An AI agent that **plays and tests your Unity game**. Give it a goal in plain
language — *"check the player can jump"* — and it observes the screen, reads
game state, sends input, watches what happens, and reports a verdict. Install
once, type `qagent` from anywhere.

```
qagent agent "check that pressing W moves the player forward"
```

## Two modes

| Mode | Command | What it is |
|---|---|---|
| **Agentic** (default) | `qagent agent "<goal>"` | An LLM autonomously drives the game in an observe → reason → act loop, like Claude Code for QA. |
| **Scripted** | `qagent run --suite x.yaml` | Deterministic YAML test suites with rule-based verification (the classic flow). |

## Features

- **One-command install** — `pip install -e .` registers `qagent` globally.
- **Autonomous agent** — LLM with vision + tool use drives the game and decides what to test next; live multi-panel dashboard shows its thoughts, actions, and findings in real time.
- **Bring any free-tier key** — Gemini, Groq, OpenRouter, Cerebras, Mistral, Together (and Anthropic). One provider abstraction; add any OpenAI-compatible host with a base URL.
- **Auto-updating model** — the agent queries the provider's `/models` endpoint at runtime and picks the newest vision+tools model. No hardcoded model ids to go stale.
- **Two input backends** — drive Unity over the **WebSocket bridge** (no focused window needed) or via **OS input** (pyautogui).
- **Screen + state observation** — WebSocket game-state JSON, screen capture, OCR (Tesseract), and OpenCV template matching.
- **Personas** — casual, speedrunner, explorer, griefer — each shapes how the agent tests (griefer actively tries to break the game).
- **Rich reports** — console panels + JSON files for CI.

---

## Quick Start

```bash
# 1. Clone & install (editable so code updates are picked up automatically)
git clone https://github.com/AshuraXX2206/unity-qa-agent.git
cd unity-qa-agent
pip install -e .          # or: py -m pip install -e .   (Windows launcher)

# 2. Configure (provider + free API key)
qagent setup

# 3. Run the agent
qagent agent "check the player can move and jump"
```

> **Windows PATH note:** pip installs `qagent.exe` into your Python's `Scripts`
> directory. If `qagent` isn't found, add that folder to PATH (the install log
> prints its location), then open a new terminal.

The setup wizard asks for:
1. **AI provider** — Gemini / Groq / OpenRouter / Cerebras / Mistral / Together / Anthropic
2. **API key** — it shows where to get a free one
3. **Model** — leave as `auto` (newest discovered at runtime) or pin one
4. **Unity project path**, **default persona**, **WebSocket URL**

Settings persist in `~/.qagent/config.yaml`.

---

## Commands

```bash
qagent                                   # interactive REPL
qagent agent "<goal>"                    # autonomous QA agent
qagent agent "<goal>" --persona griefer  # test as a griefer (tries to break things)
qagent agent "<goal>" --input bridge     # drive Unity via the WebSocket bridge
qagent agent "<goal>" --no-bridge --safe-mode   # vision/OCR only, log input without sending
qagent models                            # list models the provider offers (+ which 'auto' picks)
qagent run --suite test_cases/basic_movement.yaml --persona casual
qagent list --suite test_cases/basic_movement.yaml
qagent validate --suite test_cases/basic_movement.yaml
qagent analyze --path /path/to/Assets/Scripts     # detect input bindings & patterns
qagent generate --path /path/to/Assets/Scripts    # auto-generate a YAML suite
qagent setup                             # re-run the wizard
qagent config                            # view config
```

### `agent` flags

| Flag | Meaning |
|---|---|
| `--persona <name>` | casual / speedrunner / explorer / griefer |
| `--input {auto,bridge,os}` | input backend (default `auto`: bridge if connected, else OS) |
| `--max-steps N` | cap on agent tool-calling steps (default 15) |
| `--safe-mode` | log input actions without executing them |
| `--no-bridge` | skip the Unity bridge (observe via screen capture / OCR only) |

---

## Unity Setup

1. **Import a WebSocket library** into your Unity project:
   - **websocket-sharp** (recommended): NuGet, or drop `websocket-sharp.dll` into `Assets/Plugins/`.
   - **NativeWebSocket**: <https://github.com/endel/NativeWebSocket>.

2. **Copy** `unity-bridge/QABridge.cs` into your project (e.g. `Assets/Scripts/QA/`).

3. **Create an empty GameObject** named `QABridge`, attach the `QABridge` component, set `Player Transform` (or tag your player `Player`).

4. **Press Play** — the console prints `[QABridge] WebSocket server started on port 8765`.

### Making the agent's input reach your game

The bridge receives actions like `{"action":"key_hold","key":"w","duration":1}`.
How your game *feels* them depends on your input backend:

- **New Input System** (`com.unity.inputsystem`) — **drop-in**. `QABridge` injects
  events through `InputSystem`, so your normal `Input.GetKey` / `InputAction` /
  `PlayerInput` see simulated input with **no code changes**.
- **Legacy Input Manager** — Unity's legacy `Input` can't be injected. Read the
  bridge's simulated buffer in your input code:
  ```csharp
  // before:  if (Input.GetKey(KeyCode.W))
  // after:   if (QAInput.GetKey(KeyCode.W))     // real OR simulated
  ```
  `QAInput` (shipped in `QABridge.cs`) returns real-or-simulated input. Mouse
  clicks are additionally dispatched via `EventSystem` + `Physics.Raycast`, so UI
  buttons and world colliders react without any change.

---

## Personas

| Persona | Behaviour |
|---|---|
| `casual` | Obvious path, skips tutorials |
| `speedrunner` | Fastest, most direct route |
| `explorer` | Tries everything, pokes at edges |
| `griefer` | Actively tries to break the game; reports glitches as BUGs |

In agentic mode the persona is injected into the agent's system prompt, shaping
how it explores. In scripted mode it controls input timing and randomness.

---

## Scripted Test Suites (YAML)

Test cases live in `test_cases/`. Each file defines a **suite**.

```yaml
test_suite: "Basic Movement"
setup:
  scene: "Level_01"
  wait_for_load: 2.0
test_cases:
  - id: "TC001"
    name: "Move Forward"
    steps:
      - action: key_press
        key: "w"
        duration: 1.0
    verify:
      - type: game_state
        field: "player.position.z"
        operator: "greater_than"
        expected: 0.5
        tolerance: 0.1
```

**Actions:** `key_press`, `key_hold`, `key_combo`, `mouse_click`, `mouse_move`,
`mouse_drag`, `type_text`, `scroll`, `wait`.

**Verification types:** `game_state` (dotted JSON path + operator + tolerance),
`animation`, `ui_active`, `screen_text` (OCR), `screen_image` (template match).

---

## Reports

After each run the agent writes:
- **Console** — rich panels (agentic: verdict + activity log; scripted: pass/fail table).
- **JSON** — `reports/agent_report_<ts>.json` or `reports/report_<ts>.json`.
- **Screenshots** — `reports/screenshots/`.

---

## Project Structure

```
unity-qa-agent/
├── unity-bridge/QABridge.cs        # C# bridge: state broadcast + simulated input
├── agent/
│   ├── main.py                     # CLI + interactive REPL
│   ├── qa_agent.py                 # agentic observe→reason→act loop
│   ├── agent_tools.py              # tools the agent calls (screenshot, input, …)
│   ├── llm/                        # provider abstraction
│   │   ├── base.py                 #   neutral interface + types
│   │   ├── anthropic_provider.py   #   Claude (native SDK)
│   │   ├── gemini_provider.py      #   Gemini (native SDK)
│   │   ├── openai_compat_provider.py  # Groq/OpenRouter/Cerebras/Mistral/Together
│   │   └── registry.py             #   known providers + factory
│   ├── model_selector.py           # auto-pick newest model via /models
│   ├── bridge_client.py            # async WebSocket client (read state / send input)
│   ├── bridge_input.py             # InputExecutor that drives Unity over the bridge
│   ├── input_executor.py           # OS input via pyautogui
│   ├── screen_observer.py          # capture + OCR + template matching
│   ├── test_runner.py              # scripted YAML execution
│   ├── persona.py                  # behaviour profiles
│   ├── report_generator.py         # console + JSON reports
│   ├── tui.py                      # banner + live AgentDashboard
│   ├── config.py / setup_wizard.py # ~/.qagent/config.yaml
│   └── code_reader.py              # C# analysis → test suggestions
├── test_cases/                     # YAML suites
└── tests/                          # pytest suite (providers, agent loop, bridge round-trip)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

`tests/test_bridge_loop.py` stands up a fake `QABridge` WebSocket server and
verifies the full read-state / send-action loop without needing Unity.

---

## Troubleshooting

- **`qagent` not found** — the Python `Scripts` dir isn't on PATH; add it and open a new terminal.
- **No model / "could not discover"** — set a provider key via `qagent setup`; check `qagent models`.
- **Bridge won't connect** — ensure Unity is in Play mode, `QABridge` enabled, port `8765` open. The client retries, then falls back to screen-only mode.
- **Agent input does nothing in-game** — wire `QAInput` (legacy Input Manager) or confirm the New Input System package is installed; or use `--input os`.
- **OCR misses text** — verify `tesseract --version`; heavy font effects reduce accuracy — prefer `screen_image` template matching.

## License

MIT
