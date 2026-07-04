extends CharacterBody2D
class_name BossController

const MAX_BULLET_DETECTS: int = 3

@onready var collisions: BossCollision = $BossMechanics/Collisions as BossCollision
@onready var floating_movement: FloatingMovement = $BossMechanics/floating_movement as FloatingMovement
@onready var damage_area: DamageArea = $DamageArea as DamageArea
@onready var shot_attack: ShotAttack = $BossMechanics/ShotAttack as ShotAttack

@export_category("Boss Config")
@export var force_action_mode: bool = false
@export var action_forced: int = 1

@export_category("Boss Stats")
@export var max_health: float = 10000.0
@export var base_damage: float = 200.0
@export var damage_increment: float = 0.1
@export var max_damage: float = 1000.0
@export var max_phases: int = 2

@export_category("Movement Parameters")
@export var max_speed: float = 500.0
@export var acceleration_speed: float = 15.0 
@export var deceleration_speed: float = 10.0

var boss_actions: Dictionary = {
	0: func(d): _move_action(d),
	1: func(d): _ball_attack_action(d)
}

var current_action: int = 0
var move_dir: Vector2 = Vector2.ZERO
var shot_angle: float = 0.0
var damage: float = 0.0
var health: float = 0.0

var last_shot_impact: Vector2 = Vector2.ZERO
var last_shot_step: int = -1

var bullet_from_group: StringName = "Boss"
var bullet_to_group: StringName = "Players"
var shot_impact: Vector2 = Vector2.ZERO

#Array de tamaño fijo MAX_BULLET_DETECTS con nulls o Bullet
var bullets_detected: Array = []

var near_player: PlayerController = null
var near_bullet: Bullet = null

func init() -> BossController:
	
	health = max_health
	damage = base_damage
	
	if collisions:
		collisions.setup(self)
	if floating_movement:
		floating_movement.setup(self)
	if shot_attack:
		shot_attack.setup(self)
	if damage_area:
		damage_area.setup(self)
	
	# Inicializa con nulls para tener siempre tamaño fijo
	bullets_detected.clear()
	for _i in range(MAX_BULLET_DETECTS):
		bullets_detected.append(null)
	
	return self

# -----------------------
# --- Actualizaciones ---
# -----------------------
func update(delta, nn_outputs:Dictionary={}, player:PlayerController=null) -> void:
	_dead_if_can()
	_update_bullets_detecteds()
	near_bullet = _get_near_bullet()
	near_player = player
	
	# Actualiza los valores del jefe dependiendo de si es o no automatico.
	if not nn_outputs.is_empty(): # Si la red decide
		_update_values_by_nn(nn_outputs)
	else: # Si no decide la red.
		_update_values_by_auto()
	
	_update_action(delta)
	move_and_slide()
	
	_clamp_global_positions()

# --- Para la red neuronal ---
func _update_values_by_nn(nn_outputs: Dictionary) -> void:
	if not nn_outputs.has_all(GlobalConst.OUTPUTS_NAMES):
		push_error("boss_controller.gd | El diccionario otorgado no tiene las llaves esperadas.")
	
	var raw_dir = nn_outputs["move_dir"]
	if typeof(raw_dir) != TYPE_ARRAY or raw_dir.size() < 2 or raw_dir.size() > 2:
		push_error("boss_controller.gd | La lista 'move_dirs' debe ser un array y contener solo 2 elementos. ")
	
	# Actualza los valores del jefe
	move_dir = Vector2(raw_dir[0], raw_dir[1])
	current_action = int(nn_outputs["action"]) if not force_action_mode else action_forced
	shot_angle = nn_outputs["shot_angle"] * PI

# --- Para las automaticas ---
func _update_values_by_auto() -> void:
	pass

# --- Para la informacion de las balas y bala cercana ---
func _update_bullets_detecteds() -> void:
	# Limpia las referencias inválidas (balas que ya murieron)
	for i in range(bullets_detected.size()):
		if bullets_detected[i] != null and not is_instance_valid(bullets_detected[i]):
			bullets_detected[i] = null

# --- Para la decision de acciones ---
func _update_action(delta) -> void:
	if boss_actions.has(current_action):
		boss_actions[current_action].call(delta)

# ------------------------------
# --- Obtencion de entidades ---
# ------------------------------
func _get_near_player(players: Array[PlayerController]) -> PlayerController:
	if players.is_empty(): 
		return null
	var near: PlayerController = null
	var min_dist = INF
	for player in players:
		if is_instance_valid(player):
			var dist = global_position.distance_to(player.global_position)
			if dist < min_dist:
				min_dist = dist
				near = player
	return near

func _get_near_bullet() -> Bullet:
	var near: Bullet = null
	var min_dist = INF
	for bullet in bullets_detected:
		if is_instance_valid(bullet):
			var d = global_position.distance_to(bullet.global_position)
			if d < min_dist:
				min_dist = d
				near = bullet
	return near

# ----------------------------
# --- Lógica de Acciones -----
# ----------------------------
func _move_action(delta: float) -> void:
	if not is_instance_valid(floating_movement):
		push_error("No se encontro la instancia 'floating_movement'")
	
	if floating_movement: floating_movement.update(delta)
	damage = min(damage + damage_increment, max_damage)

func _ball_attack_action(delta: float) -> void: 
	move_dir = Vector2.ZERO
	if damage > base_damage:
		damage = base_damage
	if is_instance_valid(shot_attack):
		shot_attack.update(delta)

# ----------------
# --- Utilidad ---
# ----------------
func can_shot() -> bool:
	return current_action == 1

func _clamp_global_positions() -> void:
	global_position.x = clamp(global_position.x, 0.0, GlobalConst.game_size.x)
	global_position.y = clamp(global_position.y, 0.0, GlobalConst.game_size.y)

func _dead_if_can() -> void:
	if health <= 0:
		set_process(false)
		set_physics_process(false)
		queue_free()
