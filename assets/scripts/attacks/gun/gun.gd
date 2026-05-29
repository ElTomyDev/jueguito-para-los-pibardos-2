extends Node2D
class_name  GunController

@export_category("Bullet Settings")
@export var bullet_speed: float = 500.0
@export var bullet_life_time: float = 1.5
@export var bullet_dispersion: float = 18.0

@export_category("Gun Settings")
@export var gun_damage: float = 20.0
@export var rotation_speed: float = 50.0
@export var fire_rate: float = 0.1

@onready var shot_point: Node2D = $ShotPoint
@export var bullet: PackedScene = preload("res://assets/scenes/attacks/gun/bullet.tscn")

var character: CharacterBody2D = null
var fire_timer: float = 0.0
var total_bullet_damage: float

func setup(body:PlayerController) -> void:
	character = body

@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	view_to_mouse()
	shot_if_can(delta)

func view_to_mouse() -> void:
	Utils.view_to(global_position, get_global_mouse_position(), rotation_speed, self)

func shot_if_can(delta: float) -> void:
	if Input.is_action_pressed("shot"): # Si presiona el mouse para disparar
		fire_timer -= delta # Resta el tiempo para poder disparar
	else: # Si lo suelta
		fire_timer = fire_rate # Reinicia el tiempo
	
	if fire_timer <= 0: # Si el tiempo para disparar llega a 0 o menos.
		_shot()
		fire_timer = fire_rate

func _shot() -> void:
	var bullet_instance = bullet.instantiate()
	_set_bullet_values(bullet_instance)
	
	# Agrega la bala al 'esperado' nodo principal (en la escena main donde corre el juego).
	# Solo funciona si el arma esta UN nodo por de bajo del nodo main (main/player/gun).
	get_parent().get_parent().add_child(bullet_instance)

func _set_bullet_values(bullet_instence: Bullet) -> void:
	if !(bullet_instence is Bullet): push_error("'bullet_instance' debe ser una bala.")
	
	# Configura los valores de la bala acorde a la configuracion del arma.
	bullet_instence.global_position = shot_point.global_position
	bullet_instence.speed = bullet_speed
	bullet_instence.life_time = bullet_life_time
	bullet_instence.dispersion = bullet_dispersion
	bullet_instence.damage = _get_total_bullet_damage()
	bullet_instence.group_target = "Boss"

func _get_total_bullet_damage() -> float:
	return gun_damage if not character else gun_damage + character.damage
