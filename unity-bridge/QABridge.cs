using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// QABridge — WebSocket server that exposes game state and accepts input commands
/// from an external QA agent. Attach to any GameObject in the scene.
///
/// Dependencies: websocket-sharp (recommended) or NativeWebSocket.
///   • websocket-sharp: import via NuGet or drop the DLL into Assets/Plugins.
///   • NativeWebSocket:  https://github.com/endel/NativeWebSocket
///
/// The script uses websocket-sharp by default.  If you prefer NativeWebSocket,
/// swap the using directive and adjust the server bootstrap accordingly.
/// </summary>
[AddComponentMenu("QA/QA Bridge")]
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

    // ─── Private state ───────────────────────────────────────────────
    private WebSocketSharp.Server.WebSocketServer _server;
    private float _nextBroadcast;
    private readonly Queue<string> _incomingActions = new Queue<string>();
    private readonly object _lock = new object();

    // Cached component look-ups
    private Animator _playerAnimator;
    private Canvas[] _canvases;

    // ─── Simulated input state ───────────────────────────────────────
    private readonly Dictionary<KeyCode, float> _heldKeys = new Dictionary<KeyCode, float>();

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
        TickHeldKeys();

        if (Time.time >= _nextBroadcast)
        {
            _nextBroadcast = Time.time + broadcastInterval;
            BroadcastGameState();
        }
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
        // Minimal JSON parsing without external dependency.
        var data = JsonUtility.FromJson<ActionPayload>(json);

        switch (data.action)
        {
            case "key_press":
                SimulateKeyPress(data.key);
                break;
            case "key_hold":
                SimulateKeyHold(data.key, data.duration);
                break;
            case "mouse_click":
                SimulateMouseClick(data.x, data.y);
                break;
            default:
                Debug.LogWarning($"[QABridge] Unknown action: {data.action}");
                break;
        }
    }

    private void SimulateKeyPress(string keyName)
    {
        if (string.IsNullOrEmpty(keyName)) return;
        Debug.Log($"[QABridge] key_press → {keyName}");
        // Feed into Unity's Input system via a synthetic event or a custom input
        // buffer that your game's input layer reads.  For the legacy Input Manager,
        // the simplest approach is to set a flag that the movement script polls.
        // For the new Input System, use InputSystem.QueueStateEvent.
    }

    private void SimulateKeyHold(string keyName, float duration)
    {
        if (string.IsNullOrEmpty(keyName)) return;
        Debug.Log($"[QABridge] key_hold → {keyName} for {duration}s");
        KeyCode kc;
        if (Enum.TryParse(keyName, true, out kc))
        {
            _heldKeys[kc] = Time.time + Mathf.Max(duration, 0.05f);
        }
    }

    private void SimulateMouseClick(int x, int y)
    {
        Debug.Log($"[QABridge] mouse_click → ({x}, {y})");
        // Raycast from screen point and invoke IPointerClickHandler on hit UI, or
        // Physics.Raycast for world objects.
        var pointer = new UnityEngine.EventSystems.PointerEventData(
            UnityEngine.EventSystems.EventSystem.current)
        {
            position = new Vector2(x, y)
        };
        var results = new List<UnityEngine.EventSystems.RaycastResult>();
        UnityEngine.EventSystems.EventSystem.current.RaycastAll(pointer, results);
        foreach (var r in results)
        {
            UnityEngine.EventSystems.ExecuteEvents.Execute(
                r.gameObject,
                pointer,
                UnityEngine.EventSystems.ExecuteEvents.pointerClickHandler);
        }
    }

    private void TickHeldKeys()
    {
        var expired = new List<KeyCode>();
        foreach (var kv in _heldKeys)
        {
            if (Time.time >= kv.Value) expired.Add(kv.Key);
        }
        foreach (var k in expired) _heldKeys.Remove(k);
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

    // ─── Serialisation helpers ───────────────────────────────────────

    [Serializable]
    private class ActionPayload
    {
        public string action = "";
        public string key = "";
        public float duration = 0f;
        public int x = 0;
        public int y = 0;
    }
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
