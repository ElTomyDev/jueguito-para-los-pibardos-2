extends RefCounted
class_name NNClient

var _socket: PacketPeerUDP
var _host: String = "127.0.0.1"
var _port: int = 9999
var _timeout_ms: int = 200

var _pending: bool = false
var _last_action: Dictionary = {"move_dir": [0.0, 0.0], "shot_angle": 0.0, "action": 0}
var _request_time: int = 0

# Estadísticas de timeouts para diagnóstico
var _timeout_count: int = 0
var _total_requests: int = 0
var _log_interval: int = 500  # loguea cada N requests

func _init() -> void:
	_socket = PacketPeerUDP.new()
	_socket.connect_to_host(_host, _port)

func poll() -> void:
	if not _pending:
		return
	if Time.get_ticks_msec() - _request_time > _timeout_ms:
		_timeout_count += 1
		_pending = false
		# Log periódico para no spammear la consola
		if _total_requests > 0 and _total_requests % _log_interval == 0:
			var pct = 100.0 * _timeout_count / _total_requests
			push_warning(
                "NNClient: %d/%d timeouts (%.1f%%) — si supera 5%% revisá el servidor o aumentá _timeout_ms"
				% [_timeout_count, _total_requests, pct]
			)
		return
	if _socket.get_available_packet_count() > 0:
		var raw = _socket.get_packet()
		var response = JSON.parse_string(raw.get_string_from_utf8())
		if response:
			_last_action = response
		_pending = false

func request_action(inputs: Array, reward: float) -> void:
	if _pending:
		return
	_total_requests += 1
	var msg = JSON.stringify({
		"type": "step",
		"inputs": inputs,
		"reward": reward,
	})
	_socket.put_packet(msg.to_utf8_buffer())
	_pending = true
	_request_time = Time.get_ticks_msec()

func get_last_action() -> Dictionary:
	return _last_action

func is_busy() -> bool:
	return _pending

func get_timeout_stats() -> Dictionary:
	return {
		"timeouts": _timeout_count,
		"total": _total_requests,
		"pct": (100.0 * _timeout_count / _total_requests) if _total_requests > 0 else 0.0
	}

func notify_episode_end(episode: int, total_reward: float, final_reward: float) -> void:
	var msg = JSON.stringify({
		"type": "episode_end",
		"episode": episode,
		"total_reward": total_reward,
		"reward": final_reward
	})
	_socket.put_packet(msg.to_utf8_buffer())
	var t = Time.get_ticks_msec()
	while Time.get_ticks_msec() - t < 10000:
		if _socket.get_available_packet_count() > 0:
			var raw = _socket.get_packet()
			var resp = JSON.parse_string(raw.get_string_from_utf8())
			if resp and resp.get("type") == "episode_ready":
				return
		OS.delay_msec(5)
	push_warning("NNClient: timeout esperando episode_ready")
