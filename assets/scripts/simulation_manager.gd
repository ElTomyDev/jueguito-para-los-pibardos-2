extends Node2D

var viewport_size: Vector2

@export var player              : PackedScene
@export var boss                : PackedScene
@export var player_spawn_point  : Node2D
@export var boss_spawn_point    : Node2D
@export var random_spawns       : bool = true
@export var load_best_model     : bool = false
@export var is_new_train        : bool = false

# ---- Rewards unificados ----+
# Terminales
const REWARD_WIN             : float =  100.0
const REWARD_LOSE            : float = -100.0
const REWARD_FAST_WIN_BONUS  : float =  0.2   # por step restante al ganar
const REWARD_FAST_LOSE_BONUS : float = -0.1   # por step restante al perder rápido

# Fase 0 — Movimiento
const R_MOVING               : float =  0.1   # por moverse (velocidad > umbral)
const R_STATIC               : float = -0.2   # por estar quieto

# Fase 1 — Proximidad
const R_CLOSENESS_MAX        : float =  0.4   # escala por cercanía (0 a 0.4)
const R_TOO_FAR              : float = -0.2   # si supera MIN_DIST

# Fase 2 — Disparo y puntería
const R_AIM_MAX              : float =  0.5   # escala por ángulo (0 a 0.5)
const R_DAMAGE_DEALT         : float =  2.0   # por HP quitado al jugador
const R_DAMAGE_TAKEN         : float = -0.05  # por HP perdido (normalizado)

# Fase 3 — Esquive
const R_DODGE_BULLET         : float =  0.08  # por alejarse de bala

const MIN_PLAYER_DIST        : float = 400.0
const MIN_SPEED_THRESHOLD    : float = 20.0   # px/s mínimo para "estar en movimiento"

# ---------------
#  Nodos de la NN
# ---------------
var nn          : NeuralNetwork
var trainer     : NNTrainer
var persistence : NNPersistence

# Variables para guardar el historial del último paso e iterar el algoritmo
var last_state_activation: Dictionary = {}
var last_action_taken: int = 0
var last_boss_health: float = 0.0
var last_player_health: float = 0.0
var last_angle_error: float = 0.0
var last_dist_to_player: float = 0.0
var last_dist_to_bullet: float = 0.0

var current_episode_rewards: Array = []

var is_resetting: bool = false

func _ready() -> void:
	Engine.time_scale = 1.0
	_load_train_data()
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
	
	GlobalVars.current_step += 1
	
	# 1. Obtener entradas y ejecutar Forward Pass
	var current_inputs: Array = _get_inputs_for_nn()
	var current_activation: Dictionary = nn.forward(current_inputs)
	
	# 2. Mapear salidas del Actor a las variables globales del juego
	_update_nn_outputs(current_activation["actor_outputs"])
	
	# Determinar acción discreta tomada por la red basados en umbral de probabilidad (0.5)
	var epsilon: float = max(0.05, 0.3 - GlobalVars.current_episode * 0.0001)
	var action_taken: int
	if randf() < epsilon:
		action_taken = 1  # forzar disparo para explorar
	else:
		action_taken = 1 if current_activation["actor_outputs"][3] >= 0.5 else 0
	# 3. Calcular la recompensa del paso actual de entrenamiento
	var reward: float = _calculate_reward()
	GlobalVars.current_reward += reward
	current_episode_rewards.append(reward)
	
	# 4. Entrenar usando el paso anterior completo (si existe)
	if not last_state_activation.is_empty():
		trainer.train_step(nn, last_state_activation, current_activation, reward, false, last_action_taken)

	# Guardar estado actual como referencia histórica para el próximo cuadro
	last_state_activation = _duplicate_activation(current_activation)
	last_action_taken = action_taken
	
	if GlobalVars.boss.current_action == 1:
		GlobalVars.shot_intents += 1
	
	# 5. Comprobar condiciones de fin del episodio
	if _can_episode_end():
		_handle_episode_end()

# -----------------------------
# --- Manejo de recompensas ---
# -----------------------------
func _get_current_phase() -> int:
	var ep = GlobalVars.current_episode
	if ep >= GlobalConst.PHASE_4_START: return 3
	if ep >= GlobalConst.PHASE_3_START: return 2
	if ep >= GlobalConst.PHASE_2_START: return 1
	return 0

func _calculate_reward() -> float:
	var reward: float = 0.0
	if not is_instance_valid(GlobalVars.boss): return reward
	
	var phase: int = _get_current_phase()
	var b = GlobalVars.boss
	var p = GlobalVars.players[0] if not GlobalVars.players.is_empty() else null
	
	# --- FASE 0+: Movimiento base (siempre activo) ---
	var speed = b.velocity.length()
	if speed < MIN_SPEED_THRESHOLD:
		reward += R_STATIC
	else:
		reward += R_MOVING
	
	# --- FASE 1+: Proximidad al jugador ---
	if phase >= 1 and is_instance_valid(p):
		var dist = b.global_position.distance_to(p.global_position)
		
		if dist > MIN_PLAYER_DIST:
			reward += R_TOO_FAR
		else:
			var closeness = 1.0 - clamp(dist / MIN_PLAYER_DIST, 0.0, 1.0)
			reward += R_CLOSENESS_MAX * closeness
	
	# --- FASE 2+: Disparo, puntería y daño ---
	if phase >= 2 and is_instance_valid(p):
		# Recompensa por puntería (solo si eligió atacar)
		if b.current_action == 1:
			var ideal_angle = (p.global_position - b.global_position).angle()
			var angle_diff = abs(wrapf(ideal_angle - b.shot_angle, -PI, PI))
			reward += R_AIM_MAX * (1.0 - (angle_diff / PI))
			last_angle_error = angle_diff
		
		# Daño infligido (normalizado por max_health del jugador)
		var current_hp = p.health
		var damage_dealt = last_player_health - current_hp
		if damage_dealt > 0:
			reward += (damage_dealt / p.max_health) * R_DAMAGE_DEALT
		
		# Daño recibido (normalizado)
		var damage_taken = last_boss_health - b.health
		if damage_taken > 0:
			reward += (damage_taken / b.max_health) * R_DAMAGE_TAKEN
	
	# --- FASE 3+: Esquive ---
	if phase >= 3 and is_instance_valid(b.near_bullet):
		var dist = b.global_position.distance_to(b.near_bullet.global_position)
		if last_dist_to_bullet > 0:
			var dist_increase = dist - last_dist_to_bullet
			if dist_increase > 0:
				reward += dist_increase * R_DODGE_BULLET
		last_dist_to_bullet = dist
	elif phase < 3:
		last_dist_to_bullet = 0.0
	
	if is_instance_valid(p):
		last_player_health = p.health
	if is_instance_valid(GlobalVars.boss):
		last_boss_health = GlobalVars.boss.health
	
	return reward

func _handle_episode_end() -> void:
	# Activamos la bandera para congelar el procesamiento físico durante el cambio de escena
	is_resetting = true
	var final_reward: float = 0.0
	var steps_remaining = GlobalConst.MAX_STEP_FOR_EPISODE - GlobalVars.current_step

	if GlobalVars.players.is_empty():
		# Boss ganó
		final_reward = REWARD_WIN + steps_remaining * REWARD_FAST_WIN_BONUS
	elif is_instance_valid(GlobalVars.boss) and GlobalVars.boss.health <= 0.0:
		# Boss perdió
		final_reward = REWARD_LOSE + steps_remaining * REWARD_FAST_LOSE_BONUS
	else:
		# Timeout — se trata como derrota
		final_reward = REWARD_LOSE
	
	if not last_state_activation.is_empty():
		trainer.train_step(nn, last_state_activation, {}, final_reward, true, last_action_taken)
	
	print("Fin del Episodio: ", GlobalVars.current_episode, " | Recompensa Acumulada: ", GlobalVars.current_reward)
	
	# Guardar las recompensas de este episodio en CSV
	#_save_episode_rewards_to_csv(GlobalVars.current_episode, current_episode_rewards)
	_check_and_save_best()
	_save_train_data()
	
	GlobalVars.current_episode += 1
	
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
		for i in range(GlobalConst.INPUTS): inputs.append(0.0)
		return inputs
	
	if is_instance_valid(GlobalVars.boss):
		inputs.append_array(GlobalVars.boss.get_inputs())
	if is_instance_valid(GlobalVars.boss.near_player):
		inputs.append_array(GlobalVars.boss.near_player.get_inputs())
	
	# Garantizar que siempre devuelva exactamente (GlobalConst.INPUTS) elementos rellenando si falta algo
	while inputs.size() < GlobalConst.INPUTS:
		inputs.append(0.0)
	if inputs.size() > GlobalConst.INPUTS: 
		inputs = inputs.slice(0, GlobalConst.INPUTS)
	
	return inputs

func _update_nn_outputs(output: Array) -> void:
	if not is_instance_valid(GlobalVars.boss): return
	GlobalVars.nn_outputs["move_dir"] = [output[0], output[1]]
	GlobalVars.nn_outputs["shot_angle"] = output[2]
	GlobalVars.nn_outputs["action"] = output[3]

func _check_and_save_best() -> void:
	# Guardar siempre el último modelo 
	persistence.save_network(nn, GlobalConst.SAVE_MODEL_PATH)
	# Actualizar ventana
	GlobalVars.recent_rewards.append(GlobalVars.current_reward)
	if GlobalVars.recent_rewards.size() > GlobalConst.REWARD_WINDOW:
		GlobalVars.recent_rewards.pop_front()
	
	if GlobalVars.recent_rewards.size() < GlobalConst.REWARD_WINDOW:
		return
	
	# Calcular promedio
	var avg_reward = 0.0
	for r in GlobalVars.recent_rewards:
		avg_reward += r
	avg_reward /= GlobalVars.recent_rewards.size()
	
	# Comparar con mejor promedio
	if avg_reward > GlobalVars.best_avg_reward:
		GlobalVars.best_avg_reward = avg_reward
		GlobalVars.best_avg_episode = GlobalVars.current_episode
		persistence.save_network(nn, GlobalConst.SAVE_BEST_MODEL_PATH)
		print("Nuevo mejor promedio (últimos ", GlobalConst.REWARD_WINDOW, " episodios): ", GlobalVars.best_avg_reward, " en episodio ", GlobalVars.best_avg_episode)

# --------
# Utilidad
# --------
func _reset_episode() -> void:
	# Elimina y resetea las balas.
	for bullet in GlobalVars.bullets:
		if is_instance_valid(bullet): bullet.queue_free()
		
	
	# Elimina y resetea los jugadores.
	for p in GlobalVars.players:
		if is_instance_valid(p): p.queue_free()
	
	# Elimina y resetea el boss.
	if is_instance_valid(GlobalVars.boss): GlobalVars.boss.queue_free()
	
	last_state_activation.clear()
	current_episode_rewards.clear()
	last_angle_error = 0.0
	last_dist_to_player = 0.0
	last_dist_to_bullet = 0.0
	
	GlobalVars.boss = null
	GlobalVars.bullets.clear()
	GlobalVars.players.clear()
	GlobalVars.shot_impact = Vector2.ZERO
	GlobalVars.current_step = 0
	GlobalVars.current_reward = 0.0
	GlobalVars.nn_outputs["move_dir"] = [0.0, 0.0]
	GlobalVars.nn_outputs["shot_angle"] = 0.0
	GlobalVars.nn_outputs["action"] = 0
	_spawn_entities() # Agrega devuelta las entidades.

func _init_nn_core() -> void:
	# Instanciar el Core de Inteligencia Artificial
	nn = NeuralNetwork.new()
	trainer = NNTrainer.new()
	persistence = NNPersistence.new()
	
	# Intentar cargar pesos previos
	if load_best_model:
		persistence.load_network(nn, GlobalConst.SAVE_BEST_MODEL_PATH)
	else:
		persistence.load_network(nn, GlobalConst.SAVE_MODEL_PATH)

func _spawn_entities() -> void:
	var player_instance = player.instantiate() as PlayerController
	var boss_instance = boss.instantiate() as BossController
	viewport_size = get_viewport().get_visible_rect().size
	
	if !random_spawns:
		player_instance.global_position = player_spawn_point.global_position
		boss_instance.global_position = boss_spawn_point.global_position
	else:
		var margin: float = 200.0  # distancia mínima en píxeles
		var max_attempts: int = 20
		var player_pos: Vector2
		var boss_pos: Vector2
		for i in range(max_attempts):
			player_pos = Vector2(randf_range(0.0, viewport_size.x), randf_range(0.0, viewport_size.y))
			boss_pos = Vector2(randf_range(0.0, viewport_size.x), randf_range(0.0, viewport_size.y))
			if player_pos.distance_to(boss_pos) >= margin:
				break
		player_instance.global_position = player_pos
		boss_instance.global_position = boss_pos

	get_tree().get_root().add_child.call_deferred(player_instance)
	get_tree().get_root().add_child.call_deferred(boss_instance)

func _can_episode_end() -> bool:
	# Validación de seguridad por si el Boss es nulo en el frame actual
	if GlobalVars.players.is_empty(): return true
	if not is_instance_valid(GlobalVars.boss): return true
	if not is_instance_valid(GlobalVars.players[0]): return true
	var result = GlobalVars.current_step >= GlobalConst.MAX_STEP_FOR_EPISODE or GlobalVars.boss.health <= 0.0 or GlobalVars.players[0].health <= 0.0
	return result

func _load_train_data() -> void:
	if not is_new_train:
		var data = ExternalFileManager.read_json(GlobalConst.BEST_TRAIN_DATA_PATH)
		if data.is_empty(): 
			print("No hay informacion para cargar en el archivo:", GlobalConst.BEST_TRAIN_DATA_PATH)
			return
		
		GlobalVars.current_episode = data['episode'] if not load_best_model else data['best_avg_episode']
		GlobalVars.best_avg_reward = data['best_avg_reward']
		GlobalVars.best_avg_episode = data['best_avg_episode']

func _save_train_data() -> void:
	var data: Dictionary = {
		'episode': GlobalVars.current_episode,
		'best_avg_reward': GlobalVars.best_avg_reward,
		'best_avg_episode': GlobalVars.best_avg_episode
	}
	ExternalFileManager.save_data(data, GlobalConst.BEST_TRAIN_DATA_PATH)

func _duplicate_activation(act: Dictionary) -> Dictionary:
	var dup = {}
	for key in act.keys():
		if act[key] is Array:
			dup[key] = act[key].duplicate()
		else:
			dup[key] = act[key]
	return dup

func _save_episode_rewards_to_csv(episode: int, rewards: Array) -> void:
	
	# Abrir archivo en modo lectura/escritura para añadir líneas
	var file = FileAccess.open(GlobalConst.REWARD_CSV_PATH, FileAccess.READ_WRITE)
	if file == null:
		# Si no existe, crearlo y escribir cabecera
		file = FileAccess.open(GlobalConst.REWARD_CSV_PATH, FileAccess.WRITE)
		if file:
			file.store_line("episode,step,reward")
			file.close()
		file = FileAccess.open(GlobalConst.REWARD_CSV_PATH, FileAccess.READ_WRITE)
	
	if file == null:
		print("Error: No se pudo abrir el archivo CSV para guardar recompensas.")
		return
	
	# Ir al final del archivo
	file.seek_end()
	
	# Escribir cada paso del episodio
	for step in range(rewards.size()):
		var line = "%d,%d,%.6f\n" % [episode, step + 1, rewards[step]]
		file.store_string(line)
	
	file.close()
	print("Recompensas del episodio ", episode, " guardadas en CSV.")
