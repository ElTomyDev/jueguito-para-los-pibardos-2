extends Control


@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	if is_instance_valid(GlobalVars.boss):
		_update_distance_label($ToPlayer, GlobalVars.boss.near_player)
#	_update_distance_label($ToBullet, GlobalVars.boss.near_bullet)
	queue_redraw()

func  _draw() -> void:
	if is_instance_valid(GlobalVars.boss) and is_instance_valid(GlobalVars.boss.near_player):
		draw_line(GlobalVars.boss.global_position, GlobalVars.boss.near_player.global_position, Color(Color.RED, 0.3), 5.0)
	#if is_instance_valid(GlobalVars.boss) and is_instance_valid(GlobalVars.boss.near_bullet):
	#	draw_line(GlobalVars.boss.global_position, GlobalVars.boss.near_bullet.global_position, Color(Color.RED, 0.3), 5.0)
	
func _update_distance_label(label:Label, to:Node2D) -> void:
	if is_instance_valid(GlobalVars.boss) and is_instance_valid(to):
		var dist_to= 1.0
		if to:
			dist_to = GlobalVars.boss.global_position.distance_to(to.global_position) / get_viewport_rect().size.length()
		var x_l_pos = (GlobalVars.boss.global_position.x + to.global_position.x)/ 2
		var y_l_pos = (GlobalVars.boss.global_position.y + to.global_position.y)/ 2
		label.global_position = Vector2(x_l_pos, y_l_pos)
		label.text = "%.2f" % clamp(dist_to, 0.0, 1.0)
	else:
		label.text = ""
