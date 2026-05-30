extends CharacterBody2D
class_name BossController

@onready var floating_movement: FloatingMovement = $BossMechanics/floating_movement as FloatingMovement
@onready var damage_area: DamageArea = $DamageArea as DamageArea
@onready var ball_attack: BallAttack = $BossMechanics/BallAttack as BallAttack

@export_category("Boss Stats")
@export var health: float = 10000.0
@export var damage: float = 56.1
@export var damage_increment: float = 0.01

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

var shot_dir: Vector2
var boss_pashe: int = 0
var boss_actions: Dictionary={
	0: func(): _nothing_action(),
	1: func(): _ball_attack_action()
}
var current_action: int = 0
var total_damage: float
var move_dir: Vector2 = Vector2.ZERO
var near_player: PlayerController

func _ready() -> void:
	total_damage = bullet_damage + damage
	floating_movement.setup(self)
	ball_attack.setup(self)
	damage_area.setup(self)

func _process(delta: float) -> void:
	update_boss()

func _physics_process(delta: float) -> void:
	update_action()
	floating_movement.update(delta)
	ball_attack.update(delta)
	move_and_slide()

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

func update_boss() -> void:
	dead_if_can()
	
	# Actualiza la direccion de movimiento (entre -1 y 1) 
	# Obtiene la direccion del output de la red que es una lista [x,y] en un diccionario global
	if not GlobalVars.nn_outputs.is_empty():
		move_dir = Vector2(GlobalVars.nn_outputs['move_dir'][0], GlobalVars.nn_outputs['move_dir'][1]).normalized()
		shot_dir = Vector2(GlobalVars.nn_outputs['shot_dir'][0], GlobalVars.nn_outputs['shot_dir'][1]).normalized()
		current_action = GlobalVars.nn_outputs['current_action']
	
	near_player = _get_near_player()
	
	# Actualiza la variable global
	GlobalVars.boss_health = health

func dead_if_can() -> void:
	if health <= 0:
		queue_free()

func _get_players_stats() -> Array:
	var players_health: Array = []
	var players_positions: Array = []
	
	for player in GlobalVars.players:
		players_health.append(player.health)
		players_positions.append(player.global_position)
	
	var near_player_pos: Array = [0.0, 0.0]
	if near_player:
		near_player_pos = [near_player.global_position.x, near_player.global_position.y]
	return players_health + players_positions + near_player_pos

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
	return current_action == 1 # Dispara solo en su estado de ataque de disparo

# ----------------
# --- Acciones ---
# ----------------
func update_action() -> void:
	boss_actions[current_action].call()

func _nothing_action() -> void:
	if near_player:
		Utils.view_to(global_position, near_player.global_position, rotation_speed, self)
		damage += damage_increment # incrementa el daño

func _ball_attack_action() -> void:
	Utils.view_to(global_position, shot_dir, rotation_speed, self)
