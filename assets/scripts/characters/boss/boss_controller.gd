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
	0: func(): _nothing_action(),
	1: func(): _ball_attack_action()
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
	# 1. Leer los sensores del entorno antes de procesar
	_update_environment_sensors()
	
	# 2. Consumir las salidas que la red neuronal depositó en GlobalVars
	_read_nn_inputs()
	
	# 3. Actualizar la lógica de movimiento delegada
	floating_movement.update(delta)
	move_and_slide()
	
	# 4. Actualizar la lógica de ataque delegada
	ball_attack.update(delta)
	update_action()

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
		
	# Mapeamos "move_dir" desde el Array [x, y] de la red a un Vector2
	var nn_move = GlobalVars.nn_outputs.get("move_dir", [0.0, 0.0])
	move_dir = Vector2(nn_move[0], nn_move[1])
	
	# Mapeamos el ángulo de disparo continuo
	shot_angle = GlobalVars.nn_outputs.get("shot_angle", 0.0)
	
	# Mapeamos la acción discreta (0: nada, 1: disparar)
	current_action = GlobalVars.nn_outputs.get("current_action", 0)

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

func update_action() -> void:
	if boss_actions.has(current_action):
		boss_actions[current_action].call()

func _nothing_action() -> void:
	# El comportamiento pasivo puede incluir mirar al jugador más cercano si existe
	if is_instance_valid(near_player):
		# Podés usar tu script de Utils para manejar rotación pasiva si fuera necesario
		pass

func _ball_attack_action() -> void:
	# El ataque se procesa en combinación con el script BallAttack.
	# Podés añadir lógica estética o de fases adicionales acá.
	pass
