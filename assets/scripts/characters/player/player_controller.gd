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

var auto_shot_timer: int = 0
var auto_move_timer: int = 0
var auto_jump_timer: int = 0

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
		global_position.x / viewport_size.x,
		global_position.y / viewport_size.y,
	]

func dead_if_can() -> void:
	if health <= 0.0:
		queue_free()
		GlobalVars.players.pop_at(GlobalVars.players.find(self))

func update_automatic_mechanics(delta: float) -> void:
	if is_automatic:
		_auto_shot()
		_auto_move(delta)

func _auto_shot() -> void:
	if not is_instance_valid(GlobalVars.boss): return
	var margin = 100
	var rand_shot = randi_range(20, 50)
	if auto_shot_timer >= rand_shot:
			rand_shot = randi_range(20, 50)
			auto_move_timer = 0
	if GlobalVars.current_episode >= 0:
		if GlobalVars.current_step % rand_shot == 0:
			shot_attack._shot(Utils.view_to(
			shot_attack.global_position,
			GlobalVars.boss.global_position + Vector2(randf_range(-margin, margin), randf_range(-margin, margin)),
			100.0, shot_attack, false
			))
			auto_shot_timer += 1
	

func _auto_move(delta: float) -> void:
	var rand_move = randi_range(50, 400)
	var rand_jump = randi_range(200, 400)
	
	if GlobalVars.current_step >= 0:
		if auto_move_timer >= rand_move:
			rand_move = randi_range(50, 400)
			auto_move_timer = 0
		if auto_jump_timer >= rand_jump:
			rand_jump = randi_range(200, 400)
			auto_jump_timer = 0

		if GlobalVars.current_step % rand_move == 0:
			auto_dir = -auto_dir
		velocity.x = auto_dir * (auto_vel * delta)
		if GlobalVars.current_step % rand_jump == 0:
			velocity.y = randf_range(-200, 200)
			
		auto_move_timer += 1
		auto_jump_timer += 1
	else:
		if GlobalVars.current_step % 200 == 0:
			auto_dir = -auto_dir
		velocity.x = auto_dir * (auto_vel * delta)
		if GlobalVars.current_step % 50 == 0:
			velocity.y = randf_range(-200, 200)
		

func _on_damage_area_body_entered(bullet: Bullet) -> void:
	if is_instance_valid(bullet):
		if bullet.is_in_group("Bullets") and bullet.group_target == "Players":
			damage_area.apply_damage(bullet.damage)
			bullet.delete_bullet(self)
