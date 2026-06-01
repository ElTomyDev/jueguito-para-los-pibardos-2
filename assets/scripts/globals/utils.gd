extends Node2D

func view_to(from:Vector2, to:Vector2, rotation_speed:float=1.0, rotate_object:Node2D=null, with_lerp:bool=true, base_mirror:int=0) -> Vector2:
	var direction = to - from  # vector desde el objeto al mouse
	if rotate_object:
		var target_angle = _select_target_angle(direction, base_mirror)
		if with_lerp:
			rotate_object.rotation = lerp_angle(rotation, target_angle, rotation_speed * get_process_delta_time())
		else:
			rotate_object.rotation = target_angle
	return direction

func _select_target_angle(direction, base_mirror:int) -> float:
	var base_atan2 = atan2(direction.y, direction.x)
	var dir_dict = {
		0: base_atan2, # Derecha
		1: base_atan2 + PI / 2, # Abajo
		2: base_atan2 - PI, # Izquierda
		3: base_atan2 + PI / 2# Arriba
	}
	
	if base_mirror not in dir_dict.keys():
		push_error("el parametro 'dir' debe estar entre 0 y 3.")
	return dir_dict[base_mirror]

static func calculate_proximity_reward(distance: float, max_range: float) -> float:
	if distance > max_range:
		return 0.0
	# Retorna un valor entre 0.001 y 0.005 dependiendo de qué tan cerca esté
	return 0.005 * (1.0 - (distance / max_range))
