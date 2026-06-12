extends RefCounted
class_name NNClient

var _socket: PacketPeerUDP
var _host: String = "127.0.0.1"
var _port: int = 9999
var _timeout_ms: int = 100   # tiempo máximo para considerar respuesta válida

# Estado interno
var _pending: bool = false
var _last_action: Dictionary = {"move_dir": [0.0, 0.0], "shot_angle": 0.0, "action": 0}
var _request_time: int = 0

func _init() -> void:
	_socket = PacketPeerUDP.new()
	_socket.connect_to_host(_host, _port)

# Llamar a este método cada frame (desde _process o _physics_process)
func poll() -> void:
	if not _pending:
		return
	# Timeout: si pasó demasiado tiempo, cancelar la espera
	if Time.get_ticks_msec() - _request_time > _timeout_ms:
		_pending = false
		return
	# Verificar si llegó respuesta
	if _socket.get_available_packet_count() > 0:
		var raw = _socket.get_packet()
		var response = JSON.parse_string(raw.get_string_from_utf8())
		if response:
			_last_action = response
		_pending = false

# Enviar petición de acción (no bloqueante)
func request_action(inputs: Array, reward: float) -> void:
	if _pending:
		return   # aún esperando respuesta anterior, ignorar nueva (o podrías encolar)
	var msg = JSON.stringify({
		"type": "step",
		"inputs": inputs,
		"reward": reward,
	})
	_socket.put_packet(msg.to_utf8_buffer())
	_pending = true
	_request_time = Time.get_ticks_msec()

# Obtener la última acción recibida (devuelve inmediatamente)
func get_last_action() -> Dictionary:
	return _last_action

# Para saber si está ocupado
func is_busy() -> bool:
	return _pending

# Notificar fin de episodio (sigue bloqueante pero solo ocurre cada episodio, no afecta FPS)
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
