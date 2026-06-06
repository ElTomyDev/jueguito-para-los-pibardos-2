extends Node2D

var viewport_size: Vector2

@export var player              : PackedScene
@export var boss                : PackedScene
@export var player_spawn_point  : Node2D
@export var boss_spawn_point    : Node2D
@export var random_spawns       : bool = false
@export var load_best_model     : bool = false
@export var is_new_train        : bool = false

# Recompensas principales
const REWARD_DAMAGE_DEALT    : float =  1.0      # Por dañar al jugador (fuerte)
const REWARD_DAMAGE_RECIBE   : float = -0.1      # Por recibir daño (castigo moderado)
const REWARD_WIN_EPISODE     : float = 100.0     # Victoria (alta)
const REWARD_LOSE_EPISODE    : float = -100.0    # Derrota (fuerte castigo)
const REWARD_SURVIVE_STEP    : float =  0.0      # Paso a paso neutro

# Movimiento y posicionamiento
const REWARD_NEAR_PLAYER     : float = 0.5        # Por estar cerca del jugador
const REWARD_AWAY_PLAYER     : float = -0.2       # Por alejarse del jugador
const REWARD_FOR_STATIC      : float = -0.2       # Por moverse
const REWARD_DODGE_BULLET    : float = 0.05       # Esquivar balas enemigas
const REWARD_REDUCE_DISTANCE : float = 0.05       # Por reducir la distancia hacia el jugador

# Disparo y puntería
const REWARD_GOD_AIM         : float = 0.8        # Apuntar bien
const REWARD_BAD_AIM         : float = -0.1      # Apuntar mal (error >30°)
const REWARD_NEAR_BULLET     : float = 1.0        # Bala pasa cerca del jugador
const REWARD_FAIL_BULLET     : float = -0.05      # Bala lejos del jugador

# Acción de disparar (decision de atacar)
const REWARD_SHOOT_ACTION    : float = 0.1      # Pequeño bono por decidir disparar (evita que nunca dispare)

# Velocidad de victoria/derrota
const REWARD_FAST_PLAYER_DEAD: float = 0.2       # Bono por matar rápido
const REWARD_FAST_BOSS_DEAD  : float = -0.1      # Penalización por morir rápido

const PROXIMITY_MAX_RANGE    : float = 130.0     # Rango de recompensa para proximidad de bala
const MIN_PLAYER_DIST        : float = 500.0     # Humbral minimo para penalizar por lejania al jugador.

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
	Engine.time_scale = 2.0
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
	var action_taken: int = 1 if current_activation["actor_outputs"][3] >= 0.5 else 0
	
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

func _calculate_reward() -> float:
	var reward: float = REWARD_SURVIVE_STEP
	if not is_instance_valid(GlobalVars.boss): return reward
	
	if is_instance_valid(GlobalVars.boss) and is_instance_valid(GlobalVars.boss.near_player):
		var dist = GlobalVars.boss.global_position.distance_to(GlobalVars.boss.near_player.global_position)
		if dist > MIN_PLAYER_DIST:
			reward += REWARD_AWAY_PLAYER
	
	# Si decide disparar
	if GlobalVars.boss.current_action == 1:
		reward += REWARD_SHOOT_ACTION
	
	# Castigo por quedarse quieto
	if GlobalVars.boss.velocity == Vector2.ZERO:
		reward += REWARD_FOR_STATIC 
	
	# Recompensa por daño infligido al jugador
	var current_player_health = GlobalVars.players[0].health
	var damage_dealt = last_player_health - current_player_health
	if damage_dealt > 0:
		reward += damage_dealt * REWARD_DAMAGE_DEALT
	last_player_health = GlobalVars.players[0].health
	
	# Castigo por recibir daño el Boss
	var damage_taken = last_boss_health - GlobalVars.boss.health
	if damage_taken > 0:
		reward += damage_taken * REWARD_DAMAGE_RECIBE
	last_boss_health = GlobalVars.boss.health
	
	# Por reducir la distancia hacia el jugador
	if is_instance_valid(GlobalVars.boss) and is_instance_valid(GlobalVars.boss.near_player):
		var dist = GlobalVars.boss.global_position.distance_to(GlobalVars.boss.near_player.global_position)
		if last_dist_to_player > 0:
			var dist_reduction = last_dist_to_player - dist
			if dist_reduction > 0:
				reward += dist_reduction * REWARD_REDUCE_DISTANCE
		last_dist_to_player = dist
	
	# Recompensa por alejarse de la bala más cercana
	if is_instance_valid(GlobalVars.boss.near_bullet):
		var dist = GlobalVars.boss.global_position.distance_to(GlobalVars.boss.near_bullet.global_position)
		if last_dist_to_bullet > 0:
			var dist_increase = dist - last_dist_to_bullet   # positivo si se aleja
			if dist_increase > 0:
				reward += dist_increase * REWARD_DODGE_BULLET   # factor multiplicador
		last_dist_to_bullet = dist

# Por la bala estar cerca del jugador
	var bullets = GlobalVars.bullets
	var p = GlobalVars.players[0] if GlobalVars.players.size() > 0 else null
	if p and bullets.size() > 0:
		for b in bullets:
			if is_instance_valid(b) and b.from_group == "Boss": 
				var dist = b.global_position.distance_to(p.global_position)
				if dist <= PROXIMITY_MAX_RANGE:
					reward += REWARD_NEAR_BULLET * (1.0 - (dist / PROXIMITY_MAX_RANGE))
				else: # Si la bala esta lejos
					reward += REWARD_FAIL_BULLET
	
	# Recompensa por acercarse al jugador
	if is_instance_valid(GlobalVars.boss) and is_instance_valid(GlobalVars.boss.near_player):
		var dist = GlobalVars.boss.global_position.distance_to(GlobalVars.boss.near_player.global_position)
		var max_dist = viewport_size.length()  # o un valor fijo como 1000
		var closeness = 1.0 - clamp(dist / max_dist, 0.0, 1.0)
		reward += REWARD_NEAR_PLAYER * closeness
	
	# Recompensa por apuntar hacia el jugador (incluso si no dispara)
	if is_instance_valid(GlobalVars.boss) and is_instance_valid(GlobalVars.boss.near_player):
		var ideal_angle = (GlobalVars.boss.near_player.global_position - GlobalVars.boss.global_position).angle()
		var angle_diff = abs(wrapf(ideal_angle - GlobalVars.boss.shot_angle, -PI, PI))
		
		var aim_reward = REWARD_GOD_AIM * (1.0 - (angle_diff / PI))   # máximo +0.1
		reward += aim_reward
		# Penalización adicional si el error es grande (> 30 grados)
		if angle_diff > PI/6:
			reward += REWARD_BAD_AIM
		# Recompensa por reducir el error angular (girar en la dirección correcta)
		if last_angle_error > 0:
			var error_reduction = last_angle_error - angle_diff
			if error_reduction > 0:
				reward += error_reduction * 0.5   # premia girar hacia el jugador
		last_angle_error = angle_diff
	
	if reward > 1e6 or reward < -1e6:
		print("Reward fuera de rango: ", reward)
		reward = clamp(reward, -1000.0, 1000.0)
	
	return reward

func _handle_episode_end() -> void:
	# Activamos la bandera para congelar el procesamiento físico durante el cambio de escena
	is_resetting = true
	
	# Ejecutar un último paso final de entrenamiento avisando que done = true
	if not last_state_activation.is_empty():
		var final_reward: float = 0.0
		if GlobalVars.players.is_empty():
			final_reward += REWARD_WIN_EPISODE + (GlobalConst.MAX_STEP_FOR_EPISODE - GlobalVars.current_step) * REWARD_FAST_PLAYER_DEAD # Premio por matar al jugador
		elif (is_instance_valid(GlobalVars.boss) and GlobalVars.boss.health <= 0.0):
			final_reward += REWARD_LOSE_EPISODE  + (GlobalConst.MAX_STEP_FOR_EPISODE - GlobalVars.current_step) * REWARD_FAST_BOSS_DEAD 
		elif GlobalVars.current_step >= GlobalConst.MAX_STEP_FOR_EPISODE:
			final_reward += REWARD_LOSE_EPISODE # Castigo por no matar a tiempo
	
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
		player_instance.global_position = Vector2(randf_range(0.0, viewport_size.x), randf_range(0.0, viewport_size.y))
		boss_instance.global_position = Vector2(randf_range(0.0, viewport_size.x), randf_range(0.0, viewport_size.y))
	
	get_tree().get_root().add_child.call_deferred(player_instance)
	get_tree().get_root().add_child.call_deferred(boss_instance)

func _can_episode_end() -> bool:
	# Validación de seguridad por si el Boss es nulo en el frame actual
	if not is_instance_valid(GlobalVars.boss): 
		return true
	if not is_instance_valid(GlobalVars.players[0]): 
		return true
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
