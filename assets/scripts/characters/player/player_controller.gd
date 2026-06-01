extends CharacterBody2D
class_name PlayerController

var viewport_size: Vector2

# Obtencion, creacion e instanciacion de clases y nodos
@onready var controls: PlayerControls = $PlayerMechanics/Controls as PlayerControls
@onready var smooth_movement: PlayerSmoothMovement = $PlayerMechanics/SmoothMovement as PlayerSmoothMovement
@onready var adjustable_jump: PlayerAdjustableJump = $PlayerMechanics/AdjustableJump as PlayerAdjustableJump

@onready var damage_area: DamageArea = $DamageArea as DamageArea
@onready var gun: GunController = $Gun as GunController

@export_category("Player Stats")
@export var initial_health: float = 1000.0
@export var health: float = 0.0
@export var damage: float = 100.0

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

func _ready() -> void:
	init_values()
	controls.setup(self)
	smooth_movement.setup(self)
	adjustable_jump.setup(self)
	damage_area.setup(self)
	gun.setup(self)

@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	dead_if_can()

func _physics_process(delta: float) -> void:
	controls.update()
	smooth_movement.update(delta)
	adjustable_jump.update(delta)
	
	move_and_slide()

func init_values() -> void:
	GlobalVars.players.append(self)
	health = initial_health
	viewport_size = get_viewport().get_visible_rect().size

func stats_normalized() -> Array:
	return [
		health / initial_health,
		global_position.x / viewport_size.x,
		global_position.y / viewport_size.y
	]

func dead_if_can() -> void:
	if health <= 0.0:
		queue_free()
		GlobalVars.players.pop_at(GlobalVars.players.find(self))
