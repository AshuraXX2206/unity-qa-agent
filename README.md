# Unity QA Agent

AI-powered QA agent that automatically tests Unity games through a WebSocket bridge, screen capture, simulated input, and LLM-based verification.

## Features

- **WebSocket Bridge** — C# Unity plugin streams game state JSON; Python agent sends input commands back.
- **Screen Capture & OCR** — `mss` + `pytesseract` for reading on-screen text; OpenCV template matching for UI element detection.
- **Simulated Input** — `pyautogui` drives keyboard and mouse as if a real player is testing.
- **YAML Test Suites** — Declarative test cases with multiple verification types.
- **AI Verification** — Claude (Anthropic) vision for complex checks that rule-based logic can't cover.
- **Persona Profiles** — Casual, speedrunner, explorer, griefer — each with different timing and behaviour.
- **Rich Reports** — Coloured console output + JSON files for CI integration.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.12 recommended |
| Unity | 2022.3+ | Any render pipeline |
| Tesseract OCR | 5.x | [Install guide](https://github.com/tesseract-ocr/tesseract#installing-tesseract) |
| websocket-sharp **or** NativeWebSocket | latest | Unity WebSocket library (see Unity Setup) |

### Install Tesseract OCR

**Windows** — download installer from <https://github.com/UB-Mannheim/tesseract/wiki> and add to PATH.

**macOS** — `brew install tesseract`

**Linux** — `sudo apt-get install -y tesseract-ocr`

---

## Setup

```bash
# 1. Clone
git clone https://github.com/AshuraXX2206/unity-qa-agent.git
cd unity-qa-agent

# 2. Create virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your Anthropic API key (optional — only needed for AI verification)
```

---

## Unity Setup

1. **Import a WebSocket library** into your Unity project:
   - **websocket-sharp** (recommended): add via NuGet or drop `websocket-sharp.dll` into `Assets/Plugins/`.
   - **NativeWebSocket**: <https://github.com/endel/NativeWebSocket> — install via Unity Package Manager (git URL).

2. **Copy** `unity-bridge/QABridge.cs` into your Unity project (e.g. `Assets/Scripts/QA/`).

3. **Create an empty GameObject** in your scene, name it `QABridge`, and attach the `QABridge` component.

4. **Configure** in the Inspector:
   - `Enable Bridge` — toggle on/off.
   - `Port` — default `8765`.
   - `Broadcast Interval` — how often game state is sent (seconds).
   - `Player Transform` — drag your player GameObject here (auto-detected via `Player` tag if left empty).

5. **Press Play** in the Unity Editor — the console should print `[QABridge] WebSocket server started on port 8765`.

---

## Running Tests

```bash
# Run a full test suite
python -m agent.main run --suite test_cases/basic_movement.yaml --persona casual

# Run a single test case
python -m agent.main run --suite test_cases/basic_movement.yaml --id TC001

# Screen-capture-only mode (no Unity bridge required)
python -m agent.main run --suite test_cases/ui_flow.yaml --no-bridge

# Safe mode — log actions without actually executing input
python -m agent.main run --suite test_cases/basic_movement.yaml --safe-mode

# List all test cases in a suite
python -m agent.main list --suite test_cases/basic_movement.yaml

# Validate YAML syntax
python -m agent.main validate --suite test_cases/basic_movement.yaml
```

### Personas

| Persona | Delay | Randomness | Description |
|---|---|---|---|
| `casual` | 0.5s | 30% | Skips tutorials, rushes into action |
| `speedrunner` | 0.05s | 0% | Optimises every action for speed |
| `explorer` | 0.3s | 10% | Tries everything, goes everywhere |
| `griefer` | 0.05s | 80% | Tries to break the game |

---

## Writing Test Cases

Test cases are YAML files in `test_cases/`. Each file defines a **suite** containing one or more cases.

### Schema

```yaml
test_suite: "Suite Name"           # required
description: "What this suite tests"
setup:
  scene: "SceneName"               # expected Unity scene
  wait_for_load: 2.0               # seconds to wait before first test

test_cases:
  - id: "TC001"                    # unique identifier
    name: "Human-readable name"
    description: "What is being tested"
    ai_verify: false               # set true to use LLM verification
    steps:                         # actions to perform
      - action: key_press
        key: "w"
        duration: 1.0              # hold duration (optional)
      - action: mouse_click
        x: 540
        y: 300
        button: "left"
      - action: wait
        duration: 0.5
    verify:                        # conditions to check
      - type: game_state
        field: "player.position.z"
        operator: "greater_than"   # equals | not_equals | greater_than | less_than | gte | lte
        expected: 0.5
        tolerance: 0.1
        timeout: 5.0
```

### Action Types

| Action | Parameters | Description |
|---|---|---|
| `key_press` | `key`, `duration` (opt) | Press/hold a key |
| `key_hold` | `key`, `duration` | Hold a key for N seconds |
| `key_combo` | `keys` (list) | Press key combination |
| `mouse_click` | `x`, `y`, `button` | Click at screen position |
| `mouse_move` | `x`, `y`, `duration` | Move cursor smoothly |
| `mouse_drag` | `x1`, `y1`, `x2`, `y2` | Drag from A to B |
| `type_text` | `text` | Type a string |
| `scroll` | `x`, `y`, `amount` | Scroll mouse wheel |
| `wait` | `duration` | Pause between steps |

### Verification Types

| Type | Fields | Description |
|---|---|---|
| `game_state` | `field`, `operator`, `expected`, `tolerance` | Compare a dotted path in the game state JSON |
| `animation` | `field`, `expected` | Check current animation name |
| `ui_active` | `element` | Check if a UI element is in the `activeUI` list |
| `screen_text` | `contains` | OCR the screen and search for text |
| `screen_image` | `template`, `confidence` | OpenCV template matching |

---

## Reports

After each run the agent produces:

1. **Console output** — a rich table with pass/fail per test case.
2. **JSON file** — `reports/report_<timestamp>.json` with full details.
3. **Screenshots** — `reports/screenshots/<TC_ID>_before_*.png` and `..._after_*.png`.

---

## Project Structure

```
unity-qa-agent/
├── unity-bridge/
│   └── QABridge.cs              # C# Unity WebSocket server plugin
├── agent/
│   ├── __init__.py
│   ├── __main__.py              # python -m agent entrypoint
│   ├── main.py                  # CLI (argparse)
│   ├── bridge_client.py         # Async WebSocket client
│   ├── screen_observer.py       # Screen capture + OCR + template matching
│   ├── input_executor.py        # pyautogui wrapper with safe mode
│   ├── test_runner.py           # YAML loading, step execution, verification
│   ├── ai_verifier.py           # Claude vision verification
│   ├── persona.py               # Behaviour profiles
│   └── report_generator.py      # Rich console + JSON reports
├── test_cases/
│   ├── basic_movement.yaml      # 7 movement test cases
│   ├── combat_system.yaml       # 7 combat test cases
│   └── ui_flow.yaml             # 7 UI test cases
├── reports/                     # Generated reports (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Troubleshooting

### WebSocket won't connect

- Ensure the Unity Editor is running and you pressed **Play**.
- Confirm `QABridge` component is enabled in the Inspector.
- Check the port (`8765` by default) isn't blocked by a firewall.
- The Python client retries every 3 seconds up to 10 times, then falls back to screen-capture-only mode.

### OCR not reading text

- Verify Tesseract is installed and on your PATH: `tesseract --version`.
- Game fonts with heavy effects (glow, outline, shadow) reduce OCR accuracy — consider the `screen_image` template-matching verification type instead.
- Increase the `timeout` in the verify clause to give the text time to appear.

### pyautogui not clicking the right place

- The agent uses absolute screen coordinates — if Unity isn't running full-screen or the window moved, coordinates will be off.
- Use `--safe-mode` to log actions without executing them for debugging.

### AI verifier errors

- Ensure `ANTHROPIC_API_KEY` is set in `.env`.
- The verifier is only called when `ai_verify: true` is set on a test case or when rule-based checks are ambiguous.

---

## License

MIT
