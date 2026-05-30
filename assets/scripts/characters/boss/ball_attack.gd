extends Node2D
class_name BallAttack

var boss: BossController

var bullet: PackedScene = preload("res://assets/scenes/attacks/gun/bullet.tscn")

var fire_timer:float

func setup(body:BossController) -> void:
	boss = body
	fire_timer = boss.fire_rate

func update(delta: float) -> void:
	shot_if_can(delta)

func shot_if_can(delta: float) -> void:
	if boss.can_shot(): # Si la red decide disparar
		fire_timer -= delta # Resta el tiempo para poder disparar
		if fire_timer <= 0: # Si el tiempo para disparar llega a 0 o menos.
			_shot()
			fire_timer = boss.fire_rate

func _shot() -> void:
	var bullet_instance = bullet.instantiate()
	_set_bullet_values(bullet_instance)
	get_parent().get_parent().get_parent().add_child(bullet_instance)
	boss.damage = boss.base_damage # Reinicia el daño

func _set_bullet_values(bullet_instence: Bullet) -> void:
	if !(bullet_instence is Bullet): push_error("'bullet_instance' debe ser una bala.")
	
	bullet_instence.global_position = boss.global_position
	bullet_instence.speed = boss.bullet_speed
	bullet_instence.life_time = boss.bullet_life_time
	bullet_instence.dispersion = boss.bullet_dispersion
	bullet_instence.damage = boss.damage
	bullet_instence.group_target = "Players"
	bullet_instence.from_group = "Boss"
	bullet_instence.bullet_color = Color.BLUE
