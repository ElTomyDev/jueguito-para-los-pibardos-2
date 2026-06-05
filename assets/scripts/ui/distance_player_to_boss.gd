extends Control


@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	if is_instance_valid(GlobalVars.boss) and is_instance_valid(GlobalVars.boss.near_player):
		var dist_to_player = 1.0
		if GlobalVars.boss.near_player:
			dist_to_player = GlobalVars.boss.global_position.distance_to(GlobalVars.boss.near_player.global_position) / get_viewport_rect().size.length()
		var x_l_pos = (GlobalVars.boss.global_position.x + GlobalVars.players[0].global_position.x)/ 2
		var y_l_pos = (GlobalVars.boss.global_position.y + GlobalVars.players[0].global_position.y)/ 2
		$Label.global_position = Vector2(x_l_pos, y_l_pos)
		
		$Label.text = "%.2f" % clamp(dist_to_player, 0.0, 1.0)
	queue_redraw()

func  _draw() -> void:
	if is_instance_valid(GlobalVars.boss) and is_instance_valid(GlobalVars.boss.near_player):
		draw_line(GlobalVars.boss.global_position, GlobalVars.boss.near_player.global_position, Color(Color.RED, 0.3), 5.0)
	
