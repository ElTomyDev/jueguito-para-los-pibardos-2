extends RefCounted
class_name NNClient

var _socket: PacketPeerUDP
var _host: String = "127.0.0.1"
var _port: int = 9999
var _timeout_ms: int = 8

func _init() -> void:
	_socket = PacketPeerUDP.new()
	_socket.connect_to_host(_host, _port)

func request_action(inputs: Array, reward: float) -> Dictionary:
	var msg = JSON.stringify({
		"type": "step",
		"inputs": inputs,
		"reward": reward,
	})
	_socket.put_packet(msg.to_utf8_buffer())

	var t = Time.get_ticks_msec()
	while Time.get_ticks_msec() - t < _timeout_ms:
		if _socket.get_available_packet_count() > 0:
			var raw = _socket.get_packet()
			return JSON.parse_string(raw.get_string_from_utf8())
	# fallback si timeout
	return {"move_dir": [0.0, 0.0], "shot_angle": 0.0, "action": 0}

func notify_episode_end(episode: int, total_reward: float, final_reward: float) -> void:
	var msg = JSON.stringify({
		"type": "episode_end",
		"episode": episode,
		"total_reward": total_reward,
		"reward": final_reward
	})
	_socket.put_packet(msg.to_utf8_buffer())
