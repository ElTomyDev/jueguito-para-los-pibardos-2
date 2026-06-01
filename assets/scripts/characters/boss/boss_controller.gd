extends CharacterBody2D
class_name BossController

var viewport_size: Vector2 

@onready var floating_movement: FloatingMovement = $BossMechanics/floating_movement as FloatingMovement
@onready var damage_area: DamageArea = $DamageArea as DamageArea
@onready var ball_attack: BallAttack = $BossMechanics/BallAttack as BallAttack

@export_category("Boss Stats")
@export var initial_health: float = 10000.0
@export var base_damage = 350.0
@export var damage_increment: float = 0.1
@export var max_damage: float = 1000.0
@export var max_phases: int = 2

@export_category("Attack Config")
@export var rotation_speed: float = 5.0
@export var fire_rate: float = 0.1

@export_category("Bullet Settings")
@export var bullet_speed: float = 580.0
@export var bullet_life_time: float = 3.5
@export var bullet_dispersion: float = 8.0

@export_category("Movement Parameters")
@export var max_speed: float = 150.0
@export var acceleration_speed : float = 15.0 
@export var deceleration_speed : float  = 10.0

var boss_actions: Dictionary={
	0: func(): _nothing_action(),
	1: func(): _ball_attack_action()
}
var current_action: int = 0
var move_dir: Vector2 = Vector2.ZERO
var shot_dir: Vector2

var near_player: PlayerController
var near_bullet: Bullet

# Estadisticas
var health: float
var damage: float
var boss_pashe: int = 0

func _ready() -> void:
	init_values()
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

func stats_normalized() -> Array:
	return [
		health / initial_health,
		damage / max_damage,
		self.global_position.x / viewport_size.x,
		self.global_position.y / viewport_size.y,
		velocity.x / max_speed,  # estado físico real, no output de la red
		velocity.y / max_speed,
		shot_dir.x,
		shot_dir.y,
		float(boss_pashe) / max_phases
	] + _get_near_bullet_norm_position()

func init_values() -> void:
	viewport_size = get_viewport().get_visible_rect().size
	health = initial_health
	damage = base_damage

func update_boss() -> void:
	dead_if_can()
	
	# Actualiza la direccion de movimiento (entre -1 y 1) 
	# Obtiene la direccion del output de la red que es una lista [x,y] en un diccionario global
	if not GlobalVars.nn_outputs.is_empty():
		move_dir = Vector2(GlobalVars.nn_outputs['move_dir'][0], GlobalVars.nn_outputs['move_dir'][1]).normalized()
		shot_dir = Vector2(GlobalVars.nn_outputs['shot_dir'][0], GlobalVars.nn_outputs['shot_dir'][1]).normalized()
		current_action = GlobalVars.nn_outputs['current_action']
	
	near_bullet = _get_near_bullet()
	near_player = _get_near_player()
	
	# Actualiza la variable global
	GlobalVars.boss_health = health

func dead_if_can() -> void:
	if health <= 0:
		queue_free()

func _get_near_player() -> PlayerController:
	if GlobalVars.players.is_empty(): return null
	var near: PlayerController = null
	var min_dist = INF
	
	for player in GlobalVars.players:
		var dist = global_position.distance_to(player.global_position)
		if dist < min_dist:
			min_dist = dist
			near = player
	return near

func _get_near_bullet() -> Bullet:
	var bullets: Array = get_tree().get_nodes_in_group("Bullets")
	if bullets.is_empty(): return null
	var near: Bullet = null
	var min_dist = INF
	
	for bullet in bullets:
		if bullet.from_group == "Boss": continue
		var dist = global_position.distance_to(bullet.global_position)
		if dist < min_dist:
			min_dist = dist
			near = bullet
	return near

func _get_near_bullet_norm_position() -> Array:
	if near_bullet:
		return [
			near_bullet.global_position.x / viewport_size.x,
			near_bullet.global_position.y / viewport_size.y,
		]
	return [0.0, 0.0]

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
		damage = min(max_damage, damage + damage_increment) # incrementa el daño

func _ball_attack_action() -> void:
	var shot_point = global_position + shot_dir * 100.0
	Utils.view_to(global_position, shot_point, rotation_speed, self)
