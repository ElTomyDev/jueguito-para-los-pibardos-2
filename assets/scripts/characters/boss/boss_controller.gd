extends CharacterBody2D
class_name BossController

var viewport_size: Vector2 

const MAX_BULLET_DETECTS: int = 4
const HIT_HISTORY_SIZE = 5

@onready var floating_movement: FloatingMovement = $BossMechanics/floating_movement as FloatingMovement
@onready var damage_area: DamageArea = $DamageArea as DamageArea
@onready var shot_attack: ShotAttack = $BossMechanics/ShotAttack as ShotAttack
@onready var bullet_detector: Area2D = $BulletDetector as Area2D

@export_category("Boss Stats")
@export var max_health: float = 10000.0
@export var base_damage: float = 200.0
@export var damage_increment: float = 0.1
@export var max_damage: float = 1000.0
@export var max_phases: int = 2

@export_category("Attack Config")
@export var force_attack_mode: bool = false
@export var attack_forced: int = 1

@export_category("Movement Parameters")
@export var max_speed: float = 500.0
@export var acceleration_speed: float = 15.0 
@export var deceleration_speed: float = 10.0

var boss_actions: Dictionary = {
	0: func(_d): _nothing_action(_d),
	1: func(d): _ball_attack_action(d)
}

var hit_history: Array = []

var current_action: int = 0
var move_dir: Vector2 = Vector2.ZERO
var shot_angle: float = 0.0
var damage: float = 0.0
var health: float = 0.0

var last_shot_impact: Vector2 = Vector2.ZERO

var bullet_from_group: StringName = "Boss"
var bullet_to_group: StringName = "Players"

# bullets_detected: Array de tamaño fijo MAX_BULLET_DETECTS con nulls o Bullet
var bullets_detected: Array = []

var near_player: PlayerController = null
var near_bullet: Bullet = null

func _ready() -> void:
	init_boss()

@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	dead_if_can()

func _physics_process(delta: float) -> void:
	update_boss(delta)

func init_boss() -> void:
	health = max_health
	damage = base_damage
	viewport_size = get_viewport().get_visible_rect().size

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
	
	GlobalVars.boss = self

func update_boss(delta) -> void:
	if GlobalVars.nn_outputs.has("move_dir"):
		var raw_dir = GlobalVars.nn_outputs["move_dir"]
		if typeof(raw_dir) == TYPE_ARRAY and raw_dir.size() >= 2:
			move_dir = Vector2(raw_dir[0], raw_dir[1])
	
	if GlobalVars.nn_outputs.has("action"):
		current_action = int(GlobalVars.nn_outputs["action"]) if not force_attack_mode else attack_forced
	
	if GlobalVars.nn_outputs.has("shot_angle"):
		shot_angle = GlobalVars.nn_outputs["shot_angle"] * PI
	
	near_player = _get_near_player()
	update_bullets_norm_info()
	
	_update_action(delta)
	move_and_slide()
	
	global_position.x = clamp(global_position.x, 0.0, viewport_size.x)
	global_position.y = clamp(global_position.y, 0.0, viewport_size.y)

func get_inputs() -> Array:	
	var dist_to_player   = 1.0
	var angle_to_player  = 0.0
	var player_vel       = Vector2.ZERO
	var rel_vel          = Vector2.ZERO
	var time_since_last  = 1.0
	
	if is_instance_valid(near_player):
		dist_to_player = clamp(
			global_position.distance_to(near_player.global_position) / viewport_size.length(),
			0.0, 1.0
		)
		var to_player   = (near_player.global_position - global_position).angle()
		angle_to_player = abs(wrapf(to_player - shot_angle, -PI, PI)) / PI
		player_vel      = near_player.velocity.normalized()
		rel_vel         = near_player.velocity - velocity
	
	if shot_attack.last_shot_step > 0:
		time_since_last = clamp(
			float(GlobalVars.current_step - shot_attack.last_shot_step) / float(GlobalConst.MAX_STEP_FOR_EPISODE),
			0.0, 1.0
		)
	
	var dist_to_center = global_position.distance_to(viewport_size / 2) / viewport_size.length()

	# --- Construcción de inputs de balas APLANADOS (floats individuales) ---
	# bullets_dist:      4 floats
	# bullets_positions: 8 floats (pos_x, pos_y por cada bala)
	# bullets_velocity:  8 floats (vel_x, vel_y por cada bala)
	var b_dist: Array = []
	var b_pos:  Array = []
	var b_vel:  Array = []

	for bullet in bullets_detected:
		if is_instance_valid(bullet):
			b_dist.append(clamp(global_position.distance_to(bullet.global_position) / viewport_size.length(), 0.0, 1.0))
			b_pos.append(bullet.global_position.x / viewport_size.x)
			b_pos.append(bullet.global_position.y / viewport_size.y)
			b_vel.append(bullet.velocity.x / bullet.speed)
			b_vel.append(bullet.velocity.y / bullet.speed)
		else:
			b_dist.append(1.0)
			b_pos.append(0.0)
			b_pos.append(0.0)
			b_vel.append(0.0)
			b_vel.append(0.0)

	# 15 inputs fijos del boss
	var inputs = [
		global_position.x / viewport_size.x,
		global_position.y / viewport_size.y,
		GlobalVars.shot_impact.x / viewport_size.x,
		GlobalVars.shot_impact.y / viewport_size.y,
		velocity.x / max_speed,
		velocity.y / max_speed,
		health / max_health,
		dist_to_player,
		angle_to_player,
		rel_vel.x / max_speed,
		rel_vel.y / max_speed,
		player_vel.x,
		player_vel.y,
		time_since_last,
		dist_to_center,
	]
	# + 4 + 8 + 8 = 20 inputs de balas → total boss = 35
	inputs.append_array(b_dist)
	inputs.append_array(b_pos)
	inputs.append_array(b_vel)

	return inputs  # 35 floats

func dead_if_can() -> void:
	if health <= 0:
		set_process(false)
		set_physics_process(false)
		GlobalVars.boss = null
		queue_free()

func _get_near_player() -> PlayerController:
	if GlobalVars.players.is_empty(): 
		return null
	var near: PlayerController = null
	var min_dist = INF
	for player in GlobalVars.players:
		if is_instance_valid(player):
			var dist = global_position.distance_to(player.global_position)
			if dist < min_dist:
				min_dist = dist
				near = player
	return near

func update_bullets_norm_info() -> void:
	# Limpia las referencias inválidas (balas que ya murieron)
	for i in range(bullets_detected.size()):
		if bullets_detected[i] != null and not is_instance_valid(bullets_detected[i]):
			bullets_detected[i] = null

	# Actualiza near_bullet
	near_bullet = null
	var min_dist = INF
	for bullet in bullets_detected:
		if is_instance_valid(bullet):
			var d = global_position.distance_to(bullet.global_position)
			if d < min_dist:
				min_dist = d
				near_bullet = bullet

func can_shot() -> bool:
	return current_action == 1

# ----------------------------
# --- Lógica de Acciones -----
# ----------------------------

func _update_action(delta) -> void:
	if is_instance_valid(floating_movement):
		floating_movement.update(delta)
	if boss_actions.has(current_action):
		boss_actions[current_action].call(delta)

@warning_ignore("unused_parameter")
func _nothing_action(delta: float) -> void:
	damage += min(damage + damage_increment, max_damage)

func _ball_attack_action(delta: float) -> void:
	if damage > base_damage:
		damage = base_damage
	shot_attack.update(delta)

func register_hit(hit: bool) -> void:
	hit_history.append(1.0 if hit else 0.0)
	if hit_history.size() > HIT_HISTORY_SIZE:
		hit_history.pop_front()

func _on_damage_area_body_entered(bullet: Node2D) -> void:
	if is_instance_valid(bullet):
		if bullet.is_in_group("Bullets") and bullet.group_target == "Boss":
			damage_area.apply_damage(bullet.damage)
			bullet.delete_bullet()

func _on_bullet_detector_body_entered(bullet: Node2D) -> void:
	if not (is_instance_valid(bullet) and bullet.is_in_group("Bullets") and bullet.group_target == "Boss"):
		return
	# Busca el primer slot libre (null) para insertar
	for i in range(bullets_detected.size()):
		if bullets_detected[i] == null:
			bullets_detected[i] = bullet
			return
	# Si no hay slot libre, reemplaza el primero (FIFO)
	bullets_detected[0] = bullet

func _on_bullet_detector_body_exited(bullet: Node2D) -> void:
	if not is_instance_valid(bullet):
		return
	# CORRECCIÓN: usa asignación directa, no insert()
	var idx = bullets_detected.find(bullet)
	if idx != -1:
		bullets_detected[idx] = null

func _on_damage_area_body_shape_entered(_body_rid, _body, _body_shape_index, _local_shape_index) -> void:
	pass
