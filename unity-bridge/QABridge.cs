using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.LowLevel;
#endif

/// <summary>
/// QABridge — WebSocket server that exposes game state and accepts simulated
/// input from an external QA agent. Attach to any GameObject in the scene.
///
/// Dependencies: websocket-sharp (recommended) or NativeWebSocket.
///   • websocket-sharp: import via NuGet or drop the DLL into Assets/Plugins.
///   • NativeWebSocket:  https://github.com/endel/NativeWebSocket
///
/// ──────────────────────────────────────────────────────────────────────────
///  HOW SIMULATED INPUT REACHES YOUR GAME
/// ──────────────────────────────────────────────────────────────────────────
/// The agent sends actions like {"action":"key_hold","key":"w","duration":1.0}.
/// How your game "feels" them depends on which input backend you use:
///
///  1. NEW INPUT SYSTEM (com.unity.inputsystem)  — *drop-in*
///     Compile with ENABLE_INPUT_SYSTEM (automatic when the package is
///     installed). QABridge injects events through InputSystem, so your normal
///     `Input.GetKey`, `InputAction`, and `PlayerInput` callbacks see them with
///     NO code changes.
///
///  2. LEGACY INPUT MANAGER (UnityEngine.Input)  — *one-line change*
///     Unity's legacy Input cannot be injected. Have your movement/input code
///     read QABridge's simulated buffer instead of (or OR'd with) Input:
///
///         // before:  if (Input.GetKey(KeyCode.W))
///         // after:   if (Input.GetKey(KeyCode.W) || QABridge.SimInput.GetKey(KeyCode.W))
///
///     Or centralise it with the QAInput helper at the bottom of this file:
///         if (QAInput.GetKey(KeyCode.W)) { ... }   // real OR simulated
///
/// Mouse clicks are additionally dispatched via EventSystem + Physics raycasts,
/// so UI buttons and world colliders react even without any code change.
/// </summary>
[AddComponentMenu("QA/QA Bridge")]
[DefaultExecutionOrder(-10000)] // run before game scripts so SimInput is fresh
public class QABridge : MonoBehaviour
{
    // ─── Inspector ───────────────────────────────────────────────────
    [Header("WebSocket Server")]
    [Tooltip("Enable or disable the QA bridge at runtime.")]
    public bool enableBridge = true;

    [Tooltip("Port the WebSocket server listens on.")]
    public int port = 8765;

    [Tooltip("Interval (seconds) between game-state broadcasts.")]
    [Range(0.05f, 1f)]
    public float broadcastInterval = 0.1f;

    [Header("References (auto-detected if null)")]
    [Tooltip("The player Transform whose position is reported.")]
    public Transform playerTransform;

    // ─── Static access to simulated input ────────────────────────────
    /// <summary>Query simulated input from your game code (legacy Input Manager).</summary>
    public static SimulatedInput SimInput { get; private set; } = new SimulatedInput();

    // ─── Private state ───────────────────────────────────────────────
    private WebSocketSharp.Server.WebSocketServer _server;
    private float _nextBroadcast;
    private readonly Queue<string> _incomingActions = new Queue<string>();
    private readonly object _lock = new object();

    // Cached component look-ups
    private Animator _playerAnimator;
    private Canvas[] _canvases;

    // ─── Lifecycle ───────────────────────────────────────────────────

    private void Start()
    {
        if (!enableBridge) return;
        StartServer();
        CacheReferences();
    }

    private void Update()
    {
        if (!enableBridge) return;

        ProcessIncomingActions();
        SimInput.Tick();   // expire held keys, surface key-up events

        if (Time.time >= _nextBroadcast)
        {
            _nextBroadcast = Time.time + broadcastInterval;
            BroadcastGameState();
        }
    }

    private void LateUpdate()
    {
        // Clear one-frame edge flags after every script has read them this frame.
        SimInput.EndFrame();
    }

    private void OnDestroy()
    {
        StopServer();
    }

    private void OnApplicationQuit()
    {
        StopServer();
    }

    // ─── Server management ───────────────────────────────────────────

    private void StartServer()
    {
        try
        {
            _server = new WebSocketSharp.Server.WebSocketServer($"ws://0.0.0.0:{port}");
            _server.AddWebSocketService<QABridgeSession>("/", () =>
            {
                var session = new QABridgeSession();
                session.OnActionReceived += EnqueueAction;
                return session;
            });
            _server.Start();
            Debug.Log($"[QABridge] WebSocket server started on port {port}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"[QABridge] Failed to start server: {ex.Message}");
        }
    }

    private void StopServer()
    {
        if (_server != null)
        {
            _server.Stop();
            _server = null;
            Debug.Log("[QABridge] WebSocket server stopped");
        }
    }

    // ─── Broadcast ───────────────────────────────────────────────────

    private void BroadcastGameState()
    {
        if (_server == null) return;

        string json = BuildGameStateJson();
        foreach (var path in _server.WebSocketServices.Paths)
        {
            _server.WebSocketServices[path].Sessions.Broadcast(json);
        }
    }

    private string BuildGameStateJson()
    {
        Vector3 pos = playerTransform != null ? playerTransform.position : Vector3.zero;
        string anim = GetCurrentAnimation();
        float hp = 100f;
        float maxHp = 100f;
        bool isAlive = true;

        // Try to read HP from a component named "Health" if present
        if (playerTransform != null)
        {
            var health = playerTransform.GetComponent("Health");
            if (health != null)
            {
                var hpField = health.GetType().GetField("hp");
                var maxHpField = health.GetType().GetField("maxHp");
                var aliveField = health.GetType().GetField("isAlive");
                if (hpField != null) hp = Convert.ToSingle(hpField.GetValue(health));
                if (maxHpField != null) maxHp = Convert.ToSingle(maxHpField.GetValue(health));
                if (aliveField != null) isAlive = Convert.ToBoolean(aliveField.GetValue(health));
            }
        }

        string activeUI = BuildActiveUIJson();
        string uiText = BuildUITextJson();
        float fps = 1f / Time.unscaledDeltaTime;

        var sb = new StringBuilder(512);
        sb.Append('{');
        sb.AppendFormat("\"timestamp\":{0},", DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());
        sb.Append("\"player\":{");
        sb.AppendFormat("\"position\":{{\"x\":{0:F3},\"y\":{1:F3},\"z\":{2:F3}}},", pos.x, pos.y, pos.z);
        sb.AppendFormat("\"hp\":{0:F1},", hp);
        sb.AppendFormat("\"maxHp\":{0:F1},", maxHp);
        sb.AppendFormat("\"isAlive\":{0},", isAlive ? "true" : "false");
        sb.AppendFormat("\"currentAnimation\":\"{0}\"", anim);
        sb.Append("},");
        sb.AppendFormat("\"scene\":\"{0}\",", UnityEngine.SceneManagement.SceneManager.GetActiveScene().name);
        sb.AppendFormat("\"activeUI\":{0},", activeUI);
        sb.AppendFormat("\"uiText\":{0},", uiText);
        sb.AppendFormat("\"fps\":{0:F1}", fps);
        sb.Append('}');
        return sb.ToString();
    }

    private string GetCurrentAnimation()
    {
        if (_playerAnimator == null) return "none";
        var info = _playerAnimator.GetCurrentAnimatorClipInfo(0);
        return info.Length > 0 ? info[0].clip.name : "idle";
    }

    private string BuildActiveUIJson()
    {
        _canvases = FindObjectsOfType<Canvas>();
        var sb = new StringBuilder("[");
        bool first = true;
        foreach (var c in _canvases)
        {
            if (!c.gameObject.activeInHierarchy) continue;
            if (!first) sb.Append(',');
            sb.AppendFormat("\"{0}\"", EscapeJson(c.gameObject.name));
            first = false;
        }
        sb.Append(']');
        return sb.ToString();
    }

    private string BuildUITextJson()
    {
        var texts = FindObjectsOfType<UnityEngine.UI.Text>();
        var sb = new StringBuilder("[");
        bool first = true;
        foreach (var t in texts)
        {
            if (!t.gameObject.activeInHierarchy || string.IsNullOrEmpty(t.text)) continue;
            if (!first) sb.Append(',');
            sb.AppendFormat("\"{0}\"", EscapeJson(t.text));
            first = false;
        }
        sb.Append(']');
        return sb.ToString();
    }

    // ─── Action handling ─────────────────────────────────────────────

    private void EnqueueAction(string json)
    {
        lock (_lock)
        {
            _incomingActions.Enqueue(json);
        }
    }

    private void ProcessIncomingActions()
    {
        lock (_lock)
        {
            while (_incomingActions.Count > 0)
            {
                string raw = _incomingActions.Dequeue();
                try
                {
                    ExecuteAction(raw);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[QABridge] Failed to execute action: {ex.Message}");
                }
            }
        }
    }

    private void ExecuteAction(string json)
    {
        var data = JsonUtility.FromJson<ActionPayload>(json);

        switch (data.action)
        {
            case "key_press":
                PressKey(data.key);
                break;
            case "key_hold":
                HoldKey(data.key, data.duration);
                break;
            case "key_release":
                ReleaseKey(data.key);
                break;
            case "key_combo":
                if (data.keys != null)
                    foreach (var k in data.keys) PressKey(k);
                break;
            case "mouse_click":
                MouseClick(data.x, data.y, data.button);
                break;
            case "mouse_move":
                SimInput.SetMousePosition(data.x, data.y);
                break;
            default:
                Debug.LogWarning($"[QABridge] Unknown action: {data.action}");
                break;
        }
    }

    // Tap: register the key for a few frames so GetKey and GetKeyDown both fire.
    private void PressKey(string keyName)
    {
        if (!TryParseKey(keyName, out var kc)) return;
        Debug.Log($"[QABridge] key_press -> {kc}");
        SimInput.PressFor(kc, 0.06f);
        InjectInputSystemKey(kc, true, autoRelease: true);
    }

    private void HoldKey(string keyName, float duration)
    {
        if (!TryParseKey(keyName, out var kc)) return;
        Debug.Log($"[QABridge] key_hold -> {kc} for {duration}s");
        SimInput.PressFor(kc, Mathf.Max(duration, 0.05f));
        InjectInputSystemKey(kc, true, autoRelease: false, duration: Mathf.Max(duration, 0.05f));
    }

    private void ReleaseKey(string keyName)
    {
        if (!TryParseKey(keyName, out var kc)) return;
        Debug.Log($"[QABridge] key_release -> {kc}");
        SimInput.Release(kc);
        InjectInputSystemKey(kc, false);
    }

    private void MouseClick(int x, int y, int button)
    {
        Debug.Log($"[QABridge] mouse_click -> ({x}, {y}) btn {button}");
        SimInput.SetMousePosition(x, y);
        SimInput.PressMouseFor(button, 0.06f);

        // Also dispatch a real click so UI and world colliders react with no
        // game-side code change.
        var es = UnityEngine.EventSystems.EventSystem.current;
        if (es != null)
        {
            var pointer = new UnityEngine.EventSystems.PointerEventData(es)
            {
                position = new Vector2(x, y)
            };
            var results = new List<UnityEngine.EventSystems.RaycastResult>();
            es.RaycastAll(pointer, results);
            foreach (var r in results)
            {
                UnityEngine.EventSystems.ExecuteEvents.Execute(
                    r.gameObject, pointer,
                    UnityEngine.EventSystems.ExecuteEvents.pointerClickHandler);
            }
        }

        if (Camera.main != null)
        {
            Ray ray = Camera.main.ScreenPointToRay(new Vector3(x, y, 0));
            if (Physics.Raycast(ray, out var hit))
            {
                hit.collider.gameObject.SendMessage(
                    "OnQAClick", hit, SendMessageOptions.DontRequireReceiver);
            }
        }
    }

    private void CacheReferences()
    {
        if (playerTransform == null)
        {
            var go = GameObject.FindWithTag("Player");
            if (go != null) playerTransform = go.transform;
        }
        if (playerTransform != null)
            _playerAnimator = playerTransform.GetComponent<Animator>();
    }

    private static string EscapeJson(string s)
    {
        return s.Replace("\\", "\\\\").Replace("\"", "\\\"")
                .Replace("\n", "\\n").Replace("\r", "\\r");
    }

    // ─── Key name → KeyCode ──────────────────────────────────────────

    /// <summary>Map an agent key name (pyautogui-style) to a Unity KeyCode.</summary>
    private static bool TryParseKey(string name, out KeyCode kc)
    {
        kc = KeyCode.None;
        if (string.IsNullOrEmpty(name)) return false;
        string n = name.Trim().ToLowerInvariant();

        // single letter a–z
        if (n.Length == 1 && n[0] >= 'a' && n[0] <= 'z')
            return Enum.TryParse(n.ToUpperInvariant(), out kc);
        // single digit 0–9
        if (n.Length == 1 && n[0] >= '0' && n[0] <= '9')
            return Enum.TryParse("Alpha" + n, out kc);

        switch (n)
        {
            case "space": kc = KeyCode.Space; return true;
            case "enter":
            case "return": kc = KeyCode.Return; return true;
            case "esc":
            case "escape": kc = KeyCode.Escape; return true;
            case "tab": kc = KeyCode.Tab; return true;
            case "backspace": kc = KeyCode.Backspace; return true;
            case "delete":
            case "del": kc = KeyCode.Delete; return true;
            case "shift":
            case "shiftleft": kc = KeyCode.LeftShift; return true;
            case "shiftright": kc = KeyCode.RightShift; return true;
            case "ctrl":
            case "control":
            case "ctrlleft": kc = KeyCode.LeftControl; return true;
            case "alt":
            case "altleft": kc = KeyCode.LeftAlt; return true;
            case "up": kc = KeyCode.UpArrow; return true;
            case "down": kc = KeyCode.DownArrow; return true;
            case "left": kc = KeyCode.LeftArrow; return true;
            case "right": kc = KeyCode.RightArrow; return true;
        }
        if (n.Length >= 2 && n[0] == 'f' && int.TryParse(n.Substring(1), out int fn) && fn >= 1 && fn <= 12)
            return Enum.TryParse("F" + fn, out kc);

        // Last resort: exact KeyCode name (e.g. "LeftShift").
        return Enum.TryParse(name, true, out kc);
    }

    // ─── New Input System injection (optional) ───────────────────────

    private static void InjectInputSystemKey(KeyCode kc, bool down,
        bool autoRelease = false, float duration = 0f)
    {
#if ENABLE_INPUT_SYSTEM
        var key = ToInputSystemKey(kc);
        if (key == Key.None || Keyboard.current == null) return;
        try
        {
            using (StateEvent.From(Keyboard.current, out var eventPtr))
            {
                Keyboard.current[key].WriteValueIntoEvent(down ? 1f : 0f, eventPtr);
                InputSystem.QueueEvent(eventPtr);
            }
        }
        catch (Exception ex)
        {
            Debug.LogWarning($"[QABridge] Input System inject failed: {ex.Message}");
        }
#endif
    }

#if ENABLE_INPUT_SYSTEM
    private static Key ToInputSystemKey(KeyCode kc)
    {
        if (kc >= KeyCode.A && kc <= KeyCode.Z)
            return Key.A + (kc - KeyCode.A);
        switch (kc)
        {
            case KeyCode.Space: return Key.Space;
            case KeyCode.Return: return Key.Enter;
            case KeyCode.Escape: return Key.Escape;
            case KeyCode.Tab: return Key.Tab;
            case KeyCode.LeftShift: return Key.LeftShift;
            case KeyCode.RightShift: return Key.RightShift;
            case KeyCode.LeftControl: return Key.LeftCtrl;
            case KeyCode.LeftAlt: return Key.LeftAlt;
            case KeyCode.UpArrow: return Key.UpArrow;
            case KeyCode.DownArrow: return Key.DownArrow;
            case KeyCode.LeftArrow: return Key.LeftArrow;
            case KeyCode.RightArrow: return Key.RightArrow;
            default: return Key.None;
        }
    }
#endif

    // ─── Serialisation helpers ───────────────────────────────────────

    [Serializable]
    private class ActionPayload
    {
        public string action = "";
        public string key = "";
        public string[] keys = null;
        public float duration = 0f;
        public int x = 0;
        public int y = 0;
        public int button = 0;   // 0 = left, 1 = right, 2 = middle
    }

    // ─── Simulated input buffer ──────────────────────────────────────

    /// <summary>
    /// Frame-accurate buffer of keys/mouse the agent is "holding". Read it from
    /// your game code (legacy Input Manager) via <see cref="QABridge.SimInput"/>.
    /// GetKey is the most reliable; GetKeyDown/Up are best-effort one-frame edges.
    /// </summary>
    public class SimulatedInput
    {
        private readonly Dictionary<KeyCode, float> _heldUntil = new Dictionary<KeyCode, float>();
        private readonly HashSet<KeyCode> _down = new HashSet<KeyCode>();
        private readonly HashSet<KeyCode> _up = new HashSet<KeyCode>();
        private readonly Dictionary<int, float> _mouseUntil = new Dictionary<int, float>();
        private readonly HashSet<int> _mouseDown = new HashSet<int>();

        public Vector2 MousePosition { get; private set; }

        public bool GetKey(KeyCode kc) =>
            _heldUntil.TryGetValue(kc, out var t) && Time.time < t;
        public bool GetKey(string name) =>
            TryParseKey(name, out var kc) && GetKey(kc);
        public bool GetKeyDown(KeyCode kc) => _down.Contains(kc);
        public bool GetKeyUp(KeyCode kc) => _up.Contains(kc);
        public bool GetMouseButton(int btn) =>
            _mouseUntil.TryGetValue(btn, out var t) && Time.time < t;
        public bool GetMouseButtonDown(int btn) => _mouseDown.Contains(btn);

        internal void PressFor(KeyCode kc, float seconds)
        {
            if (!GetKey(kc)) _down.Add(kc);
            _heldUntil[kc] = Time.time + seconds;
        }

        internal void Release(KeyCode kc)
        {
            if (_heldUntil.Remove(kc)) _up.Add(kc);
        }

        internal void PressMouseFor(int btn, float seconds)
        {
            if (!GetMouseButton(btn)) _mouseDown.Add(btn);
            _mouseUntil[btn] = Time.time + seconds;
        }

        internal void SetMousePosition(int x, int y) => MousePosition = new Vector2(x, y);

        internal void Tick()
        {
            // Expire held keys → surface key-up for this frame.
            var expired = new List<KeyCode>();
            foreach (var kv in _heldUntil)
                if (Time.time >= kv.Value) expired.Add(kv.Key);
            foreach (var k in expired) { _heldUntil.Remove(k); _up.Add(k); }

            var expiredMouse = new List<int>();
            foreach (var kv in _mouseUntil)
                if (Time.time >= kv.Value) expiredMouse.Add(kv.Key);
            foreach (var b in expiredMouse) _mouseUntil.Remove(b);
        }

        internal void EndFrame()
        {
            _down.Clear();
            _up.Clear();
            _mouseDown.Clear();
        }
    }
}

/// <summary>
/// Drop-in helper: returns real OR simulated input. Replace `Input.GetKey(...)`
/// with `QAInput.GetKey(...)` in your input layer to make the QA agent able to
/// drive the game through the legacy Input Manager.
/// </summary>
public static class QAInput
{
    public static bool GetKey(KeyCode kc) =>
        Input.GetKey(kc) || QABridge.SimInput.GetKey(kc);
    public static bool GetKeyDown(KeyCode kc) =>
        Input.GetKeyDown(kc) || QABridge.SimInput.GetKeyDown(kc);
    public static bool GetKeyUp(KeyCode kc) =>
        Input.GetKeyUp(kc) || QABridge.SimInput.GetKeyUp(kc);
    public static bool GetMouseButton(int btn) =>
        Input.GetMouseButton(btn) || QABridge.SimInput.GetMouseButton(btn);
    public static bool GetMouseButtonDown(int btn) =>
        Input.GetMouseButtonDown(btn) || QABridge.SimInput.GetMouseButtonDown(btn);
    public static Vector3 mousePosition =>
        QABridge.SimInput.MousePosition != Vector2.zero
            ? (Vector3)QABridge.SimInput.MousePosition
            : Input.mousePosition;
}

/// <summary>
/// Per-connection WebSocket session.
/// </summary>
public class QABridgeSession : WebSocketSharp.Server.WebSocketBehavior
{
    public event Action<string> OnActionReceived;

    protected override void OnOpen()
    {
        Debug.Log($"[QABridge] Client connected: {Context.UserEndPoint}");
    }

    protected override void OnClose(WebSocketSharp.CloseEventArgs e)
    {
        Debug.Log($"[QABridge] Client disconnected: {e.Reason}");
    }

    protected override void OnMessage(WebSocketSharp.MessageEventArgs e)
    {
        OnActionReceived?.Invoke(e.Data);
    }

    protected override void OnError(WebSocketSharp.ErrorEventArgs e)
    {
        Debug.LogWarning($"[QABridge] WS error: {e.Message}");
    }
}
