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
@export var max_health: float = 1000.0
@export var health: float = 0.0
@export var damage: float = 500.0

@export_category("Player settings")
@export var gravity: float = ProjectSettings.get_setting("physics/2d/default_gravity")

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
	adjustable_jump.update(delta)
	shot_attack.update(delta)
	#_auto_shot()
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

func _auto_shot() -> void:
	if not is_instance_valid(GlobalVars.boss): return
	var margin = 100
	if GlobalVars.current_episode > 1200:
		if GlobalVars.current_step % 50 == 0:
			shot_attack._shot(Utils.view_to(
			shot_attack.global_position,
			GlobalVars.boss.global_position + Vector2(randf_range(-100, 100), randf_range(-100, 100)),
			100.0, shot_attack, false
			))
	elif GlobalVars.current_episode > 800:
		if GlobalVars.current_step == 200:
			shot_attack._shot(Utils.view_to(
			shot_attack.global_position,
			GlobalVars.boss.global_position + Vector2(margin, margin),
			100.0, shot_attack, false
			))
		if GlobalVars.current_step == 300:
			shot_attack._shot(Utils.view_to(
			shot_attack.global_position,
			GlobalVars.boss.global_position - Vector2(margin, margin),
			100.0, shot_attack, false
			))
		if GlobalVars.current_step == 400:
			shot_attack._shot(Utils.view_to(
			shot_attack.global_position,
			GlobalVars.boss.global_position,
			100.0, shot_attack, false
			))

func _auto_scape_from_boss_horizontal(delta: float) -> void:
	if not is_instance_valid(GlobalVars.boss): return
	dir_hor = global_position.x - GlobalVars.boss.global_position.x
