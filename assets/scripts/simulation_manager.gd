extends Node2D

var viewport_size: Vector2

@export var player             : PackedScene
@export var boss                : PackedScene
@export var player_spawn_point : Node2D
@export var boss_spawn_point    : Node2D

# ---------------
#  Nodos de la NN
# ---------------
var nn          : NeuralNetwork
var trainer     : NNTrainer
var persistence : NNPersistence

# Configuracion de pasos
const MAX_STEPS_PER_EPISODE: int = 3000 # Maximos pasos posibles
var current_step: int = 0 # Paso actual

var total_reward_step: float = 0.0

func _ready() -> void:
	_spawn_entities()

func _physics_process(_delta: float) -> void:
	pass

# ------------------------------------------------------------------
#  Mapeo de Variables de Entrada y Actualizacion de Salida de la red
# ------------------------------------------------------------------
func _get_inputs_for_nn() -> Array:
	var inputs: Array = []
	if not is_instance_valid(GlobalVars.boss):
		for i in range(11): inputs.append(0.0)
		return inputs
	
	if is_instance_valid(GlobalVars.boss):
		inputs.append(GlobalVars.boss.get_inputs())
	if is_instance_valid(GlobalVars.boss.near_player):
		inputs.append(GlobalVars.boss.near_player.get_inputs())
	
	return inputs

func _update_nn_outputs(output: Array) -> void:
	if not is_instance_valid(GlobalVars.boss): return
	
	if GlobalVars.nn_outputs.is_empty():
		GlobalVars.nn_outputs = {
			"move_dir"      : [output[0], output[1]],
			"shot_angle"    : output[2],
			"current_action": output[3],
		}
	else:
		GlobalVars.nn_outputs["move_dir"] = [output[0], output[1]]
		GlobalVars.nn_outputs["shot_angle"] = output[2]
		GlobalVars.nn_outputs["current_action"] = output[3]

# --------
# Utilidad
# --------
func _reset_episode() -> void:
	# Elimina y resetea las balas.
	for bullet in GlobalVars.bullets:
		if is_instance_valid(bullet): bullet.queue_free()
	GlobalVars.bullets.clear()
	
	# Elimina y resetea los jugadores.
	for p in GlobalVars.players:
		if is_instance_valid(p): p.queue_free()
	GlobalVars.players.clear()
	
	# Elimina y resetea el boss.
	if is_instance_valid(GlobalVars.boss): GlobalVars.boss.queue_free()
	GlobalVars.boss = null
	
	_spawn_entities() # Agrega devuelta las entidades.

func _spawn_entities() -> void:
	var player_instance = player.instantiate() as PlayerController
	var boss_instance = boss.instantiate() as BossController
	
	player_instance.global_position = player_spawn_point.global_position
	boss_instance.global_position = boss_spawn_point.global_position
	
	
	get_parent().add_child.call_deferred(player_instance)
	get_parent().add_child.call_deferred(boss_instance)

func _can_episode_end() -> bool:
	return current_step >= MAX_STEPS_PER_EPISODE or (GlobalVars.boss.health <= 0.0 or GlobalVars.players.is_empty())
