extends CharacterBody2D
class_name BossController

@onready var floating_movement: FloatingMovement = $BossMechanics/floating_movement as FloatingMovement
@onready var damage_area: DamageArea = $DamageArea as DamageArea

@export_category("Boss Stats")
@export var health: float = 10000.0
@export var damage: float = 56.1

@export_category("Attack Config")
@export var rotation_speed: float = 50.0
@export var fire_rate: float = 0.5

@export_category("Bullet Settings")
@export var bullet_speed: float = 580.0
@export var bullet_life_time: float = 3.5
@export var bullet_dispersion: float = 8.0
@export var bullet_damage: float = 13.6

@export_category("Movement Parameters")
@export var max_speed: float = 150.0
@export var acceleration_speed : float = 15.0 
@export var deceleration_speed : float  = 10.0

var shot_dir:Vector2
var boss_pashe: int = 0
var boss_states: Dictionary={}
var current_state: int = 0
var total_damage: float
var move_dir: Vector2 = Vector2.ZERO
var near_player: PlayerController

func _ready() -> void:
	total_damage = bullet_damage + damage
	floating_movement.setup(self)
	damage_area.setup(self)

func _process(delta: float) -> void:
	update_boss_values()
	_update_boss_info()
	_dead_if_can()

func _physics_process(delta: float) -> void:
	floating_movement.update(delta)
	update_actions_by_state()
	move_and_slide()

func _update_boss_info() -> void:
	GlobalVars.boss_health = health

func inputs() -> Array:
	return [
		health,
		total_damage,
		self.global_position.x,
		self.global_position.y,
		move_dir.x,
		move_dir.y,
		boss_pashe
	] + _get_players_stats()

func update_boss_values_by_nn() -> void:
	# _dead_if_can()
	# Actualiza la direccion de movimiento (entre -1 y 1) 
	# Obtiene la direccion del output de la red que es una lista [x,y] en un diccionario global
	if not GlobalVars.nn_outputs.is_empty():
		move_dir = Vector2(GlobalVars.nn_outputs['move_dir'][0], GlobalVars.nn_outputs['move_dir'][1]).normalized()
		shot_dir = Vector2(GlobalVars.nn_outputs['shot_dir'][0], GlobalVars.nn_outputs['shot_dir'][1]).normalized()
		current_state = GlobalVars.nn_outputs['current_state']

func update_boss_values() -> void:
	near_player = _get_near_player()

func update_actions_by_state() -> void:
	if current_state == 0:
		if near_player:
			Utils.view_to(global_position, near_player.global_position, rotation_speed, self)
	elif  current_state == 1:
		Utils.view_to(global_position, shot_dir, rotation_speed, self)

func _dead_if_can() -> void:
	if health <= 0:
		queue_free()

func _get_players_stats() -> Array:
	var health_players: Array = []
	var players_positions: Array = []
	
	for health_value in GlobalVars.health_players.values():
		health_players.append(health_value)
	for player in GlobalVars.players:
		players_positions.append(player.global_position)

	return health_players + players_positions + [near_player.x, near_player.y]

func _get_near_player() -> CharacterBody2D:
	if not GlobalVars.players:return
	var near: PlayerController = null
	var min_dist = INF
	
	for player in GlobalVars.players:
		var dist = global_position.distance_to(player.global_position)
		if dist < min_dist:
			min_dist = dist
			near = player
	return near

func can_shot() -> bool:
	return current_state == 1
