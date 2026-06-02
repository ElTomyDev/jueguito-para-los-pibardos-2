extends Node2D
class_name  ShotAttack

@export_category("Bullet Settings")
@export var bullet_speed: float = 500.0
@export var bullet_life_time: float = 1.5
@export var bullet_dispersion: float = 18.0

@export_category("Gun Settings")
@export var gun_damage: float = 0.0
@export var rotation_speed: float = 50.0
@export var fire_rate: float = 0.1
@export var exist_gun_sprite: bool = false
@export var is_boss: bool = false

@onready var shot_point: Node2D = $ShotPoint
@export var bullet: PackedScene = preload("res://assets/scenes/attacks/gun/bullet.tscn")

var character: CharacterBody2D = null
var gun_sprite: Sprite2D = null

var fire_timer: float = 0.0
var total_bullet_damage: float

func setup(body: CharacterBody2D) -> void:
	character = body

@warning_ignore("unused_parameter")
func update(delta: float) -> void:
	shot_if_can_and_update(delta)

func _rotate(to:Vector2) -> void:
	Utils.view_to(self.global_position, to, rotation_speed, self)

func shot_if_can_and_update(delta: float) -> void:
	if is_boss:
		fire_timer -= delta # Resta el tiempo para poder disparar
		if fire_timer <= 0: # Si el tiempo para disparar llega a 0 o menos.
			_shot()
			fire_timer = character.fire_rate
		return

	
	_rotate(get_global_mouse_position()) # Rota el arma hacia donde apunta el  mouse
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
	
	# Agrega la bala al 'esperado' nodo principal (en la escena main donde corre el juego)..
	get_tree().get_root().add_child.call_deferred(bullet_instance)

func _set_bullet_values(bullet_instence: Bullet) -> void:
	if !(bullet_instence is Bullet): push_error("'bullet_instance' debe ser una bala.")
	
	# Configura los valores de la bala acorde a la configuracion del arma.
	bullet_instence.global_position = shot_point.global_position if not is_boss else character.global_position
	bullet_instence.speed = bullet_speed
	bullet_instence.life_time = bullet_life_time
	bullet_instence.dispersion = bullet_dispersion
	bullet_instence.damage = _get_total_bullet_damage()
	bullet_instence.from_group = character.bullet_from_group
	bullet_instence.group_target = character.bullet_to_group
	if is_boss:
		bullet_instence.custom_dir = Vector2(cos(character.shot_angle), sin(character.shot_angle))


func _get_total_bullet_damage() -> float:
	return gun_damage if not character else gun_damage + character.damage
