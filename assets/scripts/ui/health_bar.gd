extends VBoxContainer
class_name HealthBar

@export var health_bar: ProgressBar = null
@export var is_boss: bool = false

func update_health_bar(boss: BossController, player:PlayerController) -> void:
	if not is_instance_valid(boss) or not is_instance_valid(player): return
	if health_bar:
		# Vida del jefe
		health_bar.max_value = boss.max_health
		health_bar.value = boss.health
		
		# Vida Del jugador
		health_bar.max_value = player.max_health
		health_bar.value = player.health
		

