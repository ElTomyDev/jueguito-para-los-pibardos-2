extends CharacterBody2D
class_name PlayerController

var viewport_size: Vector2

# Obtencion, creacion e instanciacion de clases y nodos
@onready var controls: PlayerControls = $PlayerMechanics/Controls as PlayerControls
@onready var smooth_movement: PlayerSmoothMovement = $PlayerMechanics/SmoothMovement as PlayerSmoothMovement
@onready var adjustable_jump: PlayerAdjustableJump = $PlayerMechanics/AdjustableJump as PlayerAdjustableJump
@onready var shot_attack: ShotAttack = $PlayerMechanics/Attacks/Shot as ShotAttack

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

var dir_hor: int = Vector2.AXIS_X # Vector que se encarga de manejar la direccion horizontal (-1, 0, 1)
var player_id: int = 0
var bullet_from_group: StringName = "Players" # Grupo al que pertenece la bala
var bullet_to_group: StringName = "Boss" # Target de la bala

var auto_vel: float = 1800.0
var auto_dir: int = 1

var random_states: bool = false
var current_auto_state: int = 0
var auto_fire_rate: float = 0.0
var auto_shot_timer: float = 0
var auto_move_timer: int = 0
var auto_jump_timer: int = 0

var last_shot_step: int = 0

func _ready() -> void:
	init_player()
	controls.setup(self)
	smooth_movement.setup(self)
	adjustable_jump.setup(self)
	damage_area.setup(self)
	shot_attack.setup(self)

@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	dead_if_can()

func _physics_process(delta: float) -> void:
	controls.update()
	smooth_movement.update(delta)
	shot_attack.update(delta)
	adjustable_jump.update(delta)
	update_automatic_mechanics(delta)
	move_and_slide()

func init_player() -> void:
	GlobalVars.players.append(self)
	health = max_health
	viewport_size = get_viewport().get_visible_rect().size

func get_inputs() -> Array:
	
	return [
		health / max_health,
		velocity.x / max_speed,
		velocity.y / max_speed,
		velocity.length(),
		self.is_on_floor(),
		sign(self.velocity.x),
		(GlobalVars.current_step - self.last_shot_step) / GlobalConst.MAX_STEP_FOR_EPISODE,
		global_position.x / viewport_size.x,
		global_position.y / viewport_size.y,
	]

func dead_if_can() -> void:
	if health <= 0.0:
		queue_free()
		GlobalVars.players.pop_at(GlobalVars.players.find(self))

# -----------------------------
# --- Mecanicas automaticas ---
# -----------------------------
func update_automatic_mechanics(delta: float) -> void:
	if is_automatic:
		
		update_get_state()
		_auto_shot(current_auto_state, delta)
		_auto_move(current_auto_state, delta)

func update_get_state() -> void:
	if random_states:
		if GlobalVars.current_episode % 10 == 0:
			current_auto_state = randi_range(0, 2)
	else:
		if GlobalVars.current_episode > 5000:
			current_auto_state = 2
		elif GlobalVars.current_episode > 2000:
			current_auto_state = 1
		else:
			current_auto_state = 0

func update_auto_dir() -> void:
	if self.global_position.x >= viewport_size.x - 30.0:
		auto_dir = -1
	if self.global_position.x <= 30.0:
		auto_dir = 1

# --- Disparo automatico ---
func _auto_shot(state: int, delta: float) -> void:
	auto_shot_timer += delta
	if auto_shot_timer >= auto_fire_rate:
		match state:
			0: # Juega mas chill
				_chill_shot_state()
			1:
				_normal_shot_state()
			2:
				_hard_shot_state()
		auto_shot_timer = 0.0

func _chill_shot_state() -> void:
	if is_instance_valid(GlobalVars.boss):
		auto_fire_rate = 1.0
		shot_attack._shot(
			Utils.view_to(
				shot_attack.global_position,
				GlobalVars.boss.global_position,
				100.0,
				shot_attack,
				false
			)
		)

func _normal_shot_state() -> void:
	if is_instance_valid(GlobalVars.boss):
		auto_fire_rate = 0.5
		shot_attack._shot(
			Utils.view_to(
				shot_attack.global_position,
				GlobalVars.boss.global_position + Vector2(randf_range(-100, 100), randf_range(-100, 100)),
				100.0,
				shot_attack,
				false
			)
		)

func _hard_shot_state() -> void:
	if is_instance_valid(GlobalVars.boss):
		auto_fire_rate = 0.35
		shot_attack._shot(
			Utils.view_to(
				shot_attack.global_position,
				GlobalVars.boss.global_position + Vector2(randf_range(-150, 150), randf_range(-150, 150)),
				100.0,
				shot_attack,
				false
			)
		)

# --- Movimiento automatico ---
func _auto_move(state: int, delta: float) -> void:
	update_auto_dir()
	match state:
		0: # Juega mas chill
			_chill_movement_state(delta)
		1:
			_normal_movement_state(delta)
		2:
			_hard_movement_state(delta)

func _chill_movement_state(delta: float) -> void:
	velocity.x = 4500 * delta * auto_dir

func _normal_movement_state(delta: float) -> void:
	if GlobalVars.current_step % randi_range(60,100) and self.is_on_floor():
		velocity.y -= 20000 * delta
	velocity.x = 5500 * delta * auto_dir

func _hard_movement_state(delta: float) -> void:
	if GlobalVars.current_step % randi_range(50,60)  and self.is_on_floor():
		velocity.y -= randi_range(20000, 40000) * delta
	velocity.x = 7000 * delta * auto_dir

# ------------------
# --- Colisiones ---
# ------------------
func _on_damage_area_body_entered(bullet: Bullet) -> void:
	if is_instance_valid(bullet):
		if bullet.is_in_group("Bullets") and bullet.group_target == "Players":
			damage_area.apply_damage(bullet.damage)
			bullet.delete_bullet(self)
