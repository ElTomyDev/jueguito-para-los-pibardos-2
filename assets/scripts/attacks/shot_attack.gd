extends Node2D
class_name  ShotAttack

@export_category("Bullet Settings")
@export var bullet_speed: float = 500.0
@export var bullet_life_time: float = 1.3
@export var bullet_dispersion: float = 18.0
@export var bullet_color: Color

@export_category("Gun Settings")
@export var gun_damage: float = 0.0
@export var rotation_speed: float = 50.0
@export var fire_rate: float = 0.6
@export var is_boss: bool = false

@export_category("External Nodes")
@export var bullet: PackedScene = preload("res://assets/scenes/attacks/gun/bullet.tscn")
@export var shot_point: Node2D
@export var gun_sprite: Sprite2D = null

var character: CharacterBody2D = null

var fire_timer: float = fire_rate
var total_bullet_damage: float

func setup(body: CharacterBody2D) -> void:
	character = body
	_init_boss_values()

@warning_ignore("unused_parameter")
func update(delta: float) -> void:
	boss_shot(delta)
	players_shot(delta)

func boss_shot(delta: float) -> void:
	if (character is PlayerController) or not is_instance_valid(character):
		return
	fire_timer -= delta
	_shot()

func players_shot(delta: float) -> void:
	if (character is BossController) or not is_instance_valid(character):
		return 
	
	if is_instance_valid(gun_sprite):
		Utils.view_to(self.global_position, get_global_mouse_position(), rotation_speed, self)
	
	if Input.is_action_pressed("shot"): # Si presiona el mouse para disparar
		fire_timer -= delta # Resta el tiempo para poder disparar
	_shot()


func _shot() -> void:
	if fire_timer <= 0: # Si el tiempo para disparar llega a 0 o menos.
		var bullet_instance = bullet.instantiate()
		_set_bullet_values(bullet_instance)
		
		# Agrega la bala al 'esperado' nodo principal (en la escena main donde corre el juego)..
		get_tree().get_root().add_child.call_deferred(bullet_instance)
		
		fire_timer = fire_rate

func _set_bullet_values(bullet_instence: Bullet) -> void:
	if !(bullet_instence is Bullet): push_error("'bullet_instance' debe ser una bala.")
	
	bullet_instence.global_position = shot_point.global_position if is_instance_valid(shot_point) else character.global_position
	bullet_instence.life_time = bullet_life_time
	bullet_instence.speed = bullet_speed
	bullet_instence.damage = _get_total_bullet_damage()
	bullet_instence.dispersion = bullet_dispersion
	bullet_instence.from_group = character.bullet_from_group
	bullet_instence.group_target = character.bullet_to_group
	bullet_instence.bullet_color = bullet_color
	
	if character is BossController:
		bullet_instence.custom_dir = Vector2(cos(character.shot_angle), sin(character.shot_angle))

func _init_boss_values() -> void:
	if !(character is BossController) or not is_instance_valid(character):
		return
	gun_damage = 0.0
	rotation_speed = 25.0
	fire_rate = 0.1
	
	bullet_speed = 500.0
	bullet_life_time = 1.0
	bullet_dispersion = 1.0

func _get_total_bullet_damage() -> float:
	return gun_damage if not character else gun_damage + character.damage
