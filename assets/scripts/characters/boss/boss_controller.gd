extends CharacterBody2D
class_name BossController

var viewport_size: Vector2 

# Componentes mecánicos (asegurate de que los paths a los nodos hijos sean los correctos en tu escena)
@onready var floating_movement: FloatingMovement = $BossMechanics/floating_movement as FloatingMovement
@onready var damage_area: DamageArea = $DamageArea as DamageArea
@onready var ball_attack: BallAttack = $BossMechanics/BallAttack as BallAttack

@export_category("Boss Stats")
@export var initial_health: float = 10000.0
@export var base_damage: float = 100.0
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
@export var acceleration_speed: float = 15.0 
@export var deceleration_speed: float = 10.0

# Mapeo de acciones discretas decididas por la red neuronal
var boss_actions: Dictionary = {
	0: func(_d): _nothing_action(_d),
	1: func(d): _ball_attack_action(d)
}

var current_action: int = 0
var move_dir: Vector2 = Vector2.ZERO
var shot_angle: float = 0.0
var damage: float = 0.0
var health: float = 0.0

# Referencias de entorno para los inputs de la simulación
var near_player: PlayerController = null
var near_bullet: Bullet = null

func _ready() -> void:
	damage_area.setup(self)
	init_values()

@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	dead_if_can()

func _physics_process(delta: float) -> void:
	_update_environment_sensors()
	
	_read_nn_inputs()

	floating_movement.update(delta)
	update_action(delta)
	move_and_slide()

func init_values() -> void:
	health = initial_health
	damage = base_damage
	GlobalVars.boss_health = health
	
	# Inicialización de componentes hijos
	if floating_movement:
		floating_movement.setup(self)
	if ball_attack:
		ball_attack.setup(self)

func _update_environment_sensors() -> void:
	near_bullet = _get_near_bullet()
	near_player = _get_near_player()
	
	# Sincronizar el estado de vida actual para el calculador de rewards
	GlobalVars.boss_health = health

func _read_nn_inputs() -> void:
	# Evitamos colgar el script si los outputs globales todavía no se inicializaron
	if GlobalVars.nn_outputs.is_empty():
		return
		
	self.move_dir = GlobalVars.nn_outputs.get("move_dir", Vector2.ZERO)
	self.current_action = GlobalVars.nn_outputs.get("current_action", 0)
	self.shot_angle = GlobalVars.nn_outputs.get("shot_angle", 0.0)

func dead_if_can() -> void:
	if health <= 0:
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

func _get_near_bullet() -> Bullet:
	var bullets: Array = get_tree().get_nodes_in_group("Bullets")
	if bullets.is_empty(): 
		return null
	var near: Bullet = null
	var min_dist = INF
	
	for bullet in bullets:
		if not is_instance_valid(bullet) or bullet.from_group == "Boss": 
			continue
		var dist = global_position.distance_to(bullet.global_position)
		if dist < min_dist:
			min_dist = dist
			near = bullet
	return near

func can_shot() -> bool:
	return current_action == 1

# ----------------------------
# --- Lógica de Acciones -----
# ----------------------------

func update_action(delta) -> void:
	if boss_actions.has(current_action):
		boss_actions[current_action].call(delta)

func _nothing_action(delta: float) -> void:
	# El comportamiento pasivo puede incluir mirar al jugador más cercano si existe
	if is_instance_valid(near_player):
		Utils.view_to(global_position, near_player.global_position, rotation_speed, self)


func _ball_attack_action(delta: float) -> void:
	ball_attack.update(delta)
