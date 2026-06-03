extends CharacterBody2D
class_name BossController

var viewport_size: Vector2 

# Componentes mecánicos (asegurate de que los paths a los nodos hijos sean los correctos en tu escena)
@onready var floating_movement: FloatingMovement = $BossMechanics/floating_movement as FloatingMovement
@onready var damage_area: DamageArea = $DamageArea as DamageArea
@onready var shot_attack: ShotAttack = $BossMechanics/ShotAttack as ShotAttack

@export_category("Boss Stats")
@export var max_health: float = 10000.0
@export var base_damage: float = 100.0
@export var damage_increment: float = 0.1
@export var max_damage: float = 1000.0
@export var max_phases: int = 2

@export_category("Attack Config")
@export var force_attack_mode: bool = true
@export var attack_forced: int = 1
@export var rotation_speed: float = 15.0
@export var fire_rate: float = 0.3

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

var last_shot_impact: Vector2 = Vector2.ZERO

var bullet_from_group: StringName = "Boss"
var bullet_to_group: StringName = "Players"

# Referencias de entorno para los inputs de la simulación
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
	# Inicialización de componentes hijos
	if floating_movement:
		floating_movement.setup(self)
	if shot_attack:
		shot_attack.setup(self)
	if damage_area:
		damage_area.setup(self)
	
	GlobalVars.boss = self

func update_boss(delta) -> void:
	# Actualiza valores en base al output de la red.
	if GlobalVars.nn_outputs.has("move_dir"):
		var raw_dir = GlobalVars.nn_outputs["move_dir"]
		if typeof(raw_dir) == TYPE_ARRAY and raw_dir.size() >= 2:
			move_dir = Vector2(raw_dir[0], raw_dir[1])
	if GlobalVars.nn_outputs.has("current_action"):
		current_action = int(GlobalVars.nn_outputs["current_action"]) if not force_attack_mode else attack_forced
	if GlobalVars.nn_outputs.has("shot_angle"):
		shot_angle = GlobalVars.nn_outputs["shot_angle"] * PI
	
	near_bullet = _get_near_bullet()
	near_player = _get_near_player()
	
	if is_instance_valid(floating_movement):
		floating_movement.update(delta)
	_update_action(delta)
	move_and_slide()
	
	# Clamp para mantener al boss dentro del viewport
	global_position.x = clamp(global_position.x, 0.0, viewport_size.x)
	global_position.y = clamp(global_position.y, 0.0, viewport_size.y)
	

func get_inputs() -> Array:
	var near_bullet_pos = Vector2.ZERO
	if is_instance_valid(near_bullet):
		near_bullet_pos = near_bullet.global_position
	return [
		self.global_position.x / viewport_size.x,
		self.global_position.y / viewport_size.y,
		near_bullet_pos.x / viewport_size.x,
		near_bullet_pos.y / viewport_size.y,
		GlobalVars.shot_impact.x / viewport_size.x,
		GlobalVars.shot_impact.y / viewport_size.y,
		self.velocity.x / max_speed,
		self.velocity.y / max_speed,
		self.health / max_health,
	]

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

func _get_near_bullet() -> Bullet:
	var bullets: Array = GlobalVars.bullets
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

func _update_action(delta) -> void:
	if boss_actions.has(current_action):
		boss_actions[current_action].call(delta)

@warning_ignore("unused_parameter")
func _nothing_action(delta: float) -> void:
	# El comportamiento pasivo puede incluir mirar al jugador más cercano si existe
	if is_instance_valid(near_player):
		Utils.view_to(global_position, near_player.global_position, rotation_speed, self)
		
	damage += damage_increment # incrementa el daño al no atacar

func _ball_attack_action(delta: float) -> void:
	# Resetea el daño despues de acumularlo al no atacar
	if damage > base_damage:
		damage = base_damage
	shot_attack.update(delta)
