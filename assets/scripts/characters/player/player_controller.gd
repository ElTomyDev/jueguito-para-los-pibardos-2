extends CharacterBody2D
class_name PlayerController

# Obtencion, creacion e instanciacion de clases y nodos
@onready var controls: PlayerControls = $PlayerMechanics/Controls as PlayerControls
@onready var smooth_movement: PlayerSmoothMovement = $PlayerMechanics/SmoothMovement as PlayerSmoothMovement
@onready var adjustable_jump: PlayerAdjustableJump = $PlayerMechanics/AdjustableJump as PlayerAdjustableJump
@onready var shot_attack: ShotAttack = $PlayerMechanics/Attacks/Shot as ShotAttack
@onready var collisions: PlayerCollision = $PlayerMechanics/Collisions as PlayerCollision

@onready var damage_area: DamageArea = $DamageArea as DamageArea

@export_category("Player Stats")
@export var max_health: float = 500.0
@export var health: float = 0.0
@export var damage: float = 1000.0

@export_category("Player settings")
@export var gravity: float = ProjectSettings.get_setting("physics/2d/default_gravity")
@export var is_automatic: bool = true

@export_category("Movement Parameters")
@export var max_speed : float = 300.0
@export var acceleration_speed : float = 15.0 
@export var deceleration_speed : float  = 10.0 

@export_category("Jump Parameters")
@export var jump_force: float = 4000.0    # Fuerza inicial de salto (altura maxima posible)
@export var max_jump_time: float = 0.1   # Es el tiempo maximo que se puede mantener apretrada la tecla de salto.

var player_difficulty = 0.0
var dir_hor: int = Vector2.AXIS_X # Vector que se encarga de manejar la direccion horizontal (-1, 0, 1)
var player_id: int = 0
var bullet_from_group: StringName = "Players" # Grupo al que pertenece la bala
var bullet_to_group: StringName = "Boss" # Target de la bala
var shot_impact: Vector2 = Vector2.ZERO

var near_bullet: Bullet = null
var near_boss: BossController = null

var auto_vel: float = 1800.0
var auto_dir: int = 1

var random_states: bool = false
var current_auto_state: int = 0
var auto_fire_rate: float = 0.0
var auto_shot_timer: float = 0
var auto_move_timer: int = 0
var auto_jump_timer: int = 0

var last_shot_step: int = 0

func update(delta: float, boss: BossController, bullets: Array[Bullet], current_episode: int) -> void:
	near_boss = boss
	dead_if_can()
	controls.update()
	smooth_movement.update(delta)
	shot_attack.update(delta)
	adjustable_jump.update(delta)
	update_automatic_mechanics(delta, near_boss, bullets, current_episode)
	move_and_slide()

func init() -> PlayerController:
	health = max_health
	controls.setup(self)
	smooth_movement.setup(self)
	adjustable_jump.setup(self)
	damage_area.setup(self)
	shot_attack.setup(self)
	collisions.setup(self)
	return self

# MOVER A OTRO LADO
func get_inputs() -> Array:
	
	return [
		health / max_health,
		float(self.is_on_floor()),
		clamp(float(GlobalVars.current_step - self.last_shot_step) / float(GlobalConst.MAX_STEP_FOR_EPISODE), 0.0, 1.0),
		global_position.x / GlobalConst.game_size.x,
		global_position.y / GlobalConst.game_size.y,
	]

func dead_if_can() -> void:
	if health <= 0.0:
		queue_free()

# -----------------------------
# --- Mecanicas automaticas ---
# -----------------------------
func update_automatic_mechanics(delta: float, boss: BossController, bullets: Array[Bullet], current_episode: int) -> void:
	if is_automatic:
		update_get_state(current_episode)
		_auto_shot(boss, current_auto_state, delta)
		_auto_move(delta, boss, bullets)

func update_get_state(current_episode: int) -> void:
	if random_states:
		if current_episode % 10 == 0:
			current_auto_state = randi_range(0, 2)
	else:
		var hard_prob = player_difficulty
		var roll = randf()
		if roll < hard_prob * 0.6:
			current_auto_state = 2
		elif roll < hard_prob:
			current_auto_state = 1
		else:
			current_auto_state = 0

func update_auto_dir(boss: BossController) -> void:
	if not is_instance_valid(boss): return
	
	if is_instance_valid(near_bullet):
		var bullet_pos = near_bullet.global_position
		if bullet_pos.x < self.global_position.x:
			auto_dir = 1
		if bullet_pos.x > self.global_position.x:
			auto_dir = -1
	else:
		if self.global_position.distance_to(boss.global_position) > 450.0:
			if boss.global_position.x < self.global_position.x:
				auto_dir = -1
			if boss.global_position.x > self.global_position.x:
				auto_dir = 1
		elif self.global_position.distance_to(boss.global_position) < 200.0:
			if boss.global_position.x < self.global_position.x:
				auto_dir = 1
			if boss.global_position.x > self.global_position.x:
				auto_dir = -1
		else:
			auto_dir = 0

func update_difficulty(boss_win_rate_window: Array) -> void:
	# Calcula win rate del boss_scene en la ventana
	var wins = boss_win_rate_window.count(true)
	var win_rate = float(wins) / float(boss_win_rate_window.size())
	
	# Solo sube la dificultad si el boss_scene gana más del N%
	if win_rate > 0.19:
		player_difficulty = clamp(player_difficulty + 0.02, 0.0, 1.0)
	elif win_rate < 0.09:
		player_difficulty = clamp(player_difficulty - 0.01, 0.0, 1.0)

# --- Disparo automatico ---
func _auto_shot(boss: BossController, state: int, delta: float) -> void:
	auto_shot_timer += delta
	if auto_shot_timer >= auto_fire_rate:
		match state:
			0: # Juega mas chill
				_chill_shot_state(boss)
			1:
				_normal_shot_state(boss)
			2:
				_hard_shot_state(boss)
		auto_shot_timer = 0.0

func _chill_shot_state(boss: BossController) -> void:
	if is_instance_valid(boss) and randf() < 0.25:
		auto_fire_rate = 0.5
		shot_attack._shot(
			Utils.view_to(
				shot_attack.global_position,
				boss.global_position,
				100.0,
				shot_attack,
				false
			)
		)

func _normal_shot_state(boss: BossController) -> void:
	if is_instance_valid(boss) and randf() < 0.50:
		auto_fire_rate = 0.3
		shot_attack._shot(
			Utils.view_to(
				shot_attack.global_position,
				boss.global_position + Vector2(randf_range(-100, 100), randf_range(-100, 100)),
				100.0,
				shot_attack,
				false
			)
		)

func _hard_shot_state(boss: BossController) -> void:
	if is_instance_valid(boss) and randf() < 0.85:
		auto_fire_rate = 0.2
		shot_attack._shot(
			Utils.view_to(
				shot_attack.global_position,
				boss.global_position + Vector2(randf_range(-150, 150), randf_range(-150, 150)),
				100.0,
				shot_attack,
				false
			)
		)

# --- Movimiento automatico ---
func _auto_move(delta: float, boss: BossController, bullets: Array[Bullet]) -> void:
	if not is_instance_valid(boss): return
	near_bullet = _get_near_bullet(bullets)
	update_auto_dir(boss)
	if not is_instance_valid(near_bullet) or self.global_position.distance_to(near_bullet.global_position) > 65.0 or randf() > player_difficulty:
		
		velocity.x = 7000 * delta * auto_dir
	else:
		velocity.x = 7000 * delta * auto_dir
		if global_position.distance_to(near_bullet.global_position) <= 30.0 and self.is_on_floor() and near_bullet.global_position.y > self.global_position.y - 10:
			velocity.y -= randi_range(20000, 40000) * delta

func _get_near_bullet(bullets: Array[Bullet]) -> Bullet:
	if bullets.is_empty(): return null
	var n_bullet = null
	
	var min_dist = INF
	for bullet in bullets:
		if bullet.from_group == "Boss":
			var d = global_position.distance_to(bullet.global_position)
			if d < min_dist:
				min_dist = d
				n_bullet = bullet
	return n_bullet

# ------------------
# --- Colisiones ---
# ------------------
