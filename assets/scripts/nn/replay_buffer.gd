extends RefCounted
class_name ReplayBuffer

var capacity: int
var buffer: Array = []
var position: int = 0

func _init(cap: int = 512) -> void:
	capacity = cap

func add(state_act: Dictionary, next_state_act: Dictionary, reward: float, done: bool, action: int) -> void:
	var transition = {
		"state": state_act,
		"next": next_state_act,
		"reward": reward,
		"done": done,
		"action": action
	}
	if buffer.size() < capacity:
		buffer.append(transition)
	else:
		buffer[position] = transition
	position = (position + 1) % capacity

func sample(batch_size: int) -> Array:
	var n = min(batch_size, buffer.size())
	var indices = []
	while indices.size() < n:
		var idx = randi() % buffer.size()
		if not idx in indices:
			indices.append(idx)
	var batch = []
	for i in indices:
		batch.append(buffer[i])
	return batch

func size() -> int:
	return buffer.size()

func is_ready(min_size: int) -> bool:
	return buffer.size() >= min_size
