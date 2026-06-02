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

# Variables para guardar el historial del último paso e iterar el algoritmo
var last_state_activation: Dictionary = {}
var last_action_taken: int = 0
var last_boss_health: float = 0.0
var last_player_health: float = 0.0

var is_resetting: bool = false

func _ready() -> void:
	_init_nn_core()
	_spawn_entities()

func _physics_process(_delta: float) -> void:
	# Si estamos en medio de un reset, o las entidades aún no se registraron en el árbol, ignoramos el frame.
	# Esto evita por completo el spawn múltiple e infinito de jefes.
	if is_resetting:
		return
	 # Esperamos a que ambas entidades estén registradas antes de simular
	if not is_instance_valid(GlobalVars.boss) or GlobalVars.players.is_empty():
		return
	if is_instance_valid(GlobalVars.boss):
		last_boss_health = GlobalVars.boss.max_health # O 10000.0 directamente
	
	current_step += 1
	
	# 1. Obtener entradas y ejecutar Forward Pass
	var current_inputs: Array = _get_inputs_for_nn()
	var current_activation: Dictionary = nn.forward(current_inputs)
	
	# 2. Mapear salidas del Actor a las variables globales del juego
	_update_nn_outputs(current_activation["actor_outputs"])
	
	# Determinar acción discreta tomada por la red basados en umbral de probabilidad (0.5)
	var action_taken: int = 1 if current_activation["actor_outputs"][3] >= 0.5 else 0
	
	# 3. Calcular la recompensa del paso actual de entrenamiento
	var reward: float = _calculate_reward()
	total_reward_step += reward
	
	# 4. Entrenar usando el paso anterior completo (si existe)
	if not last_state_activation.is_empty():
		trainer.train_step(nn, last_state_activation, current_activation, reward, false, last_action_taken)
		
	# Guardar estado actual como referencia histórica para el próximo cuadro
	last_state_activation = current_activation
	last_action_taken = action_taken
	
	# 5. Comprobar condiciones de fin del episodio
	if _can_episode_end():
		_handle_episode_end()

func _calculate_reward() -> float:
	var r: float = 0.0
	
	# Recompensa por supervivencia temporal básica
	r += 0.01 
	
	# Recompensa/Castigo basado en la variación de vidas
	if is_instance_valid(GlobalVars.boss):
		var boss_damage_taken: float = last_boss_health - GlobalVars.boss.health
		if boss_damage_taken > 0.0:
			r -= (boss_damage_taken * 0.1) # Penalización severa por dejarse pegar
		last_boss_health = GlobalVars.boss.health

	if not GlobalVars.players.is_empty() and is_instance_valid(GlobalVars.players[0]):
		var player_damage_taken: float = last_player_health - GlobalVars.players[0].health
		if player_damage_taken > 0.0:
			r += (player_damage_taken * 0.2) # Recompensa alta por dañar al jugador
		last_player_health = GlobalVars.players[0].health
		
	return r

func _handle_episode_end() -> void:
	# Activamos la bandera para congelar el procesamiento físico durante el cambio de escena
	is_resetting = true
	
	# Guardamos de forma segura las variables para el reporte antes de limpiarlas
	var steps_saved: int = current_step
	
	# Ejecutar un último paso final de entrenamiento avisando que done = true
	if not last_state_activation.is_empty():
		var final_reward: float = 0.0
		if GlobalVars.players.is_empty():
			final_reward += 10.0 # Premio gordo por matar al jugador
		elif is_instance_valid(GlobalVars.boss) and GlobalVars.boss.health <= 0.0:
			final_reward -= 10.0 # Castigo gordo por morir
			
		trainer.train_step(nn, last_state_activation, {}, final_reward, true, last_action_taken)
	
	# Guardar cerebro actualizado al disco de forma persistente
	persistence.save_network(nn)
	
	# Reiniciar métricas de la simulación
	print("Fin del Episodio. Pasos totales: ", steps_saved, " | Recompensa Acumulada: ", total_reward_step)
	total_reward_step = 0.0
	last_state_activation.clear()
	
	_reset_episode()
	_reset_health_tracking()

func _reset_health_tracking() -> void:
	# Esperamos dos cuadros de física obligatorios para dar tiempo real a que queue_free limpie 
	# y a que call_deferred registre las nuevas entidades en GlobalVars
	await get_tree().physics_frame
	await get_tree().physics_frame
	
	if is_instance_valid(GlobalVars.boss):
		last_boss_health = GlobalVars.boss.health
	if not GlobalVars.players.is_empty() and is_instance_valid(GlobalVars.players[0]):
		last_player_health = GlobalVars.players[0].health
		
	# Una vez sincronizadas las variables de vida, liberamos el manager para simular el próximo episodio
	is_resetting = false

# ------------------------------------------------------------------
#  Mapeo de Variables de Entrada y Actualizacion de Salida de la red
# ------------------------------------------------------------------
func _get_inputs_for_nn() -> Array:
	var inputs: Array = []
	if not is_instance_valid(GlobalVars.boss):
		for i in range(11): inputs.append(0.0)
		return inputs
	
	if is_instance_valid(GlobalVars.boss):
		inputs.append_array(GlobalVars.boss.get_inputs())
	if is_instance_valid(GlobalVars.boss.near_player):
		inputs.append_array(GlobalVars.boss.near_player.get_inputs())
	
	# Garantizar que siempre devuelva exactamente 11 elementos rellenando si falta algo
	while inputs.size() < 11:
		inputs.append(0.0)
	if inputs.size() > 11: # Si tiene mas de 11 elementos
		inputs = inputs.slice(0, 11)
	
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

func _init_nn_core() -> void:
	# Instanciar el Core de Inteligencia Artificial
	nn = NeuralNetwork.new()
	trainer = NNTrainer.new()
	persistence = NNPersistence.new()
	
	# Intentar cargar pesos previos
	persistence.load_network(nn)

func _spawn_entities() -> void:
	var player_instance = player.instantiate() as PlayerController
	var boss_instance = boss.instantiate() as BossController
	
	player_instance.global_position = player_spawn_point.global_position
	boss_instance.global_position = boss_spawn_point.global_position
	
	get_tree().get_root().add_child.call_deferred(player_instance)
	get_tree().get_root().add_child.call_deferred(boss_instance)
	
	current_step = 0

func _can_episode_end() -> bool:
	# Validación de seguridad por si el Boss es nulo en el frame actual
	if not is_instance_valid(GlobalVars.boss): return true
	return current_step >= MAX_STEPS_PER_EPISODE or GlobalVars.boss.health <= 0.0 or GlobalVars.players.is_empty()
