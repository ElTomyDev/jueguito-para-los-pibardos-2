extends VBoxContainer
class_name HealthBar

@export var health_bar: ProgressBar = null
@export var is_boss: bool = false

func _process(delta: float) -> void:
	update_health_bar()

func update_health_bar() -> void:
	if health_bar:
		if _is_boss_health_bar():
			health_bar.max_value = GlobalVars.boss.max_health
			health_bar.value = GlobalVars.boss.health
		elif _is_player_health_bar():
			health_bar.max_value = GlobalVars.players[0].max_health
			health_bar.value = GlobalVars.players[0].health
		else:
			return

func _is_boss_health_bar() -> bool:
	return is_boss and is_instance_valid(GlobalVars.boss)

func _is_player_health_bar() -> bool:
	return !is_boss and (not GlobalVars.players.is_empty())
