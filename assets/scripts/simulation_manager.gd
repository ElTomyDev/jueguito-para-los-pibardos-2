extends Node2D

var viewport_size: Vector2

@export var force_attack_mode: bool = true # Permitir que la IA decida si dispara o no
@export var force_attack: int = 1 
@export var players             : Array[PackedScene]
@export var spawn_player_points : Array[Node2D]
@export var boss_spawn          : Node2D
@export var boss                : PackedScene

# ─────────────────────────────────────────
#  Sistema de recompensas
# ─────────────────────────────────────────
const REWARD_DAMAGE_DEALT    : float =  5.0    # Por dañar al jugador
const REWARD_SURVIVE_STEP    : float =  0.01   # Por sobrevivir un paso
const REWARD_WIN_EPISODE     : float = 100.0   # Por ganar la partida
const REWARD_LOSE_EPISODE    : float = -150.0  # Por perden la partida
const REWARD_DODGE_BULLET    : float = 0.009   # Por esquivar balas
const REWARD_DAMAGE_RECIBE   : float = -5.0    # Por recibir daño
const REWARD_STATIC_VELOCITY : float = -0.02   # Por quedarse quieto

# ─────────────────────────────────────────
#  Nodos de la NN
# ─────────────────────────────────────────
var nn          : NeuralNetwork
var trainer     : NNTrainer
var persistence : NNPersistence

var boss_instance: BossController = null
var current_step: int = 0 # Paso actual
const MAX_STEPS_PER_EPISODE: int = 3000 # Maximos pasos posibles

# Control de estado de salud para recompensas delta
var last_boss_health: float = 0.0
var last_player_health: float = 0.0

func _ready() -> void:
	viewport_size = get_viewport_rect().size
	
	# Instanciar componentes del sistema de IA
	nn = NeuralNetwork.new()
	add_child(nn)
	
	trainer = NNTrainer.new()
	add_child(trainer)
	trainer.setup(nn)
	
	persistence = NNPersistence.new()
	add_child(persistence)
	
	# Intentar cargar cerebro previo
	if not persistence.load_into(nn, trainer):
		print("SimulationManager: Inicializando nuevo cerebro aleatorio.")
		
	_reset_simulation()

func _physics_process(_delta: float) -> void:
	if not is_instance_valid(boss_instance) or not _any_player_alive():
		_handle_episode_end()
		return
		
	current_step += 1
	if current_step >= MAX_STEPS_PER_EPISODE:
		_handle_episode_end()
		return
		
	# 1. Obtener estado del entorno (Inputs)
	var inputs: Array = _get_neural_network_inputs()
	
	# 2. Forward pass de la red (Obtenemos las medias de la política)
	var raw_output: Array = nn.forward(inputs)
	if raw_output.is_empty(): return
	
	# 3. Aplicar exploración estocástica (Ruido Gaussiano/Uniforme basado en current_sigma)
	var sigma: float = trainer.current_sigma
	var action: Array = []
	
	# Outputs continuos (0, 1, 2) -> Añadir ruido de exploración
	for i in range(3):
		var noise: float = randf_range(-sigma, sigma)
		action.append(clampf(raw_output[i] + noise, -1.0, 1.0))
		
	# Output discreto (3) -> Probabilidad de disparo (Bernoulli)
	# Guardamos la acción discreta final tomada (0 o 1)
	var shoot_prob: float = raw_output[3]
	var shoot_action: float = 1.0 if randf() < shoot_prob else 0.0
	action.append(shoot_action)
	
	# 4. Aplicar acciones físicas al Boss
	_apply_action_to_boss(action)
	
	# 5. Calcular recompensa obtenida en este step
	var reward: float = _calculate_reward()
	
	# 6. Registrar el paso en el buffer del entrenador
	trainer.record_step(inputs, raw_output, action, reward)
	
	# Actualizar estados de salud locales para el próximo step
	last_boss_health = boss_instance.health if is_instance_valid(boss_instance) else 0.0
	last_player_health = _get_total_player_health()

func _handle_episode_end() -> void:
	# Determinar recompensa final del episodio
	var final_reward: float = 0.0
	if is_instance_valid(boss_instance) and boss_instance.health > 0:
		if not _any_player_alive():
			final_reward = REWARD_WIN_EPISODE
	else:
		final_reward = REWARD_LOSE_EPISODE
		
	if not trainer.episode_rewards.is_empty():
		trainer.episode_rewards[trainer.episode_rewards.size() - 1] += final_reward
		
	# Ejecutar optimización por REINFORCE
	print("Finalizando Episodio ", trainer.episode_count, " - Pasos: ", current_step, " - Sigma: ", trainer.current_sigma)
	trainer.end_episode()
	
	# Guardar progreso periódicamente
	if trainer.episode_count % 10 == 0:
		persistence.save(nn, trainer)
		
	_reset_simulation()

# ─────────────────────────────────────────
#  Mapeo de Variables de Entrada (15 Inputs)
# ─────────────────────────────────────────
func _get_neural_network_inputs() -> Array:
	var inputs: Array = []
	if not is_instance_valid(boss_instance):
		for i in range(15): inputs.append(0.0)
		return inputs
	
	# Posición relativa y estado del Boss
	inputs.append(boss_instance.global_position.x / viewport_size.x)
	inputs.append(boss_instance.global_position.y / viewport_size.y)
	inputs.append(boss_instance.velocity.x / boss_instance.max_speed)
	inputs.append(boss_instance.velocity.y / boss_instance.max_speed)
	inputs.append(boss_instance.health / boss_instance.initial_health)
	
	# Relación con el Jugador más cercano
	var near_player = boss_instance.near_player
	if is_instance_valid(near_player):
		var dir_to_p = (near_player.global_position - boss_instance.global_position)
		inputs.append(dir_to_p.x / viewport_size.x)
		inputs.append(dir_to_p.y / viewport_size.y)
		inputs.append(near_player.velocity.x / near_player.max_speed)
		inputs.append(near_player.velocity.y / near_player.max_speed)
		inputs.append(near_player.health / near_player.initial_health)
	else:
		for i in range(5): inputs.append(0.0)
		
	# Relación con la Bala enemiga más cercana (Peligro)
	var near_bullet = boss_instance.near_bullet
	if is_instance_valid(near_bullet):
		var dir_to_b = (near_bullet.global_position - boss_instance.global_position)
		inputs.append(dir_to_b.x / viewport_size.x)
		inputs.append(dir_to_b.y / viewport_size.y)
		inputs.append(near_bullet.global_position.x)
		inputs.append(near_bullet.global_position.y)
		inputs.append(near_bullet.speed / 1000.0)
	else:
		for i in range(5): inputs.append(0.0)
		
	return inputs

# ─────────────────────────────────────────
#  Aplicación de salidas de la red al entorno
# ─────────────────────────────────────────
func _apply_action_to_boss(action: Array) -> void:
	if not is_instance_valid(boss_instance): return
	
	var shot_angle : float = action[2] * PI
	GlobalVars.nn_outputs = {
		"move_dir"      : [action[0], action[1]],
		"shot_angle"    : shot_angle,
		"current_action": force_attack if force_attack_mode else int(action[3]),
	}

# ─────────────────────────────────────────
#  Cálculo de Recompensas por Step
# ─────────────────────────────────────────
func _calculate_reward() -> float:
	var reward: float = REWARD_SURVIVE_STEP
	if not is_instance_valid(boss_instance): return reward
	
	# 1. Recompensa por daño infligido al jugador
	var current_player_health = _get_total_player_health()
	var damage_dealt = last_player_health - current_player_health
	if damage_dealt > 0:
		reward += damage_dealt * REWARD_DAMAGE_DEALT
		
	# 2. Castigo por recibir daño el Boss
	var damage_taken = last_boss_health - boss_instance.health
	if damage_taken > 0:
		reward += damage_taken * REWARD_DAMAGE_RECIBE
		
	# 3. Penalización por velocidad nula (evitar parálisis)
	if boss_instance.velocity.length() < 5.0:
		reward += REWARD_STATIC_VELOCITY
		
	# 4. Incentivo por esquivar balas cercanas
	if is_instance_valid(boss_instance.near_bullet):
		var dist = boss_instance.global_position.distance_to(boss_instance.near_bullet.global_position)
		if dist > 150.0 and dist < 300.0:
			reward += REWARD_DODGE_BULLET
			
	return reward

# ─────────────────────────────────────────
#  Reinicio total de la escena de simulación
# ─────────────────────────────────────────
func _reset_simulation() -> void:
	current_step = 0
	
	# 1. Limpiar proyectiles existentes
	var bullets = get_tree().get_nodes_in_group("Bullets")
	for b in bullets:
		if is_instance_valid(b): b.queue_free()
		
	# 2. Eliminar jugadores viejos y respawnear nuevos
	for p in GlobalVars.players:
		if is_instance_valid(p): p.queue_free()
	GlobalVars.players.clear()
	
	for i in range(players.size()):
		var p_inst = players[i].instantiate()
		p_inst.global_position = spawn_player_points[i].global_position
		get_parent().add_child.call_deferred(p_inst)
		GlobalVars.players.append(p_inst)
		
	# 3. Reiniciar por completo al Boss
	if is_instance_valid(boss_instance):
		boss_instance.queue_free()
		
	boss_instance = boss.instantiate() as BossController
	boss_instance.global_position = boss_spawn.global_position
	# Forzamos la restauración de los puntos de vida iniciales
	boss_instance.health = boss_instance.initial_health 
	get_parent().add_child.call_deferred(boss_instance)
	
	GlobalVars.boss_health = boss_instance.initial_health
	
	# Inicializar deltas de salud
	last_boss_health = boss_instance.initial_health
	last_player_health = _get_total_player_health()

func _get_total_player_health() -> float:
	var total : float = 0.0
	for p in GlobalVars.players:
		if is_instance_valid(p):
			total += p.health
	return total

func _any_player_alive() -> bool:
	for p in GlobalVars.players:
		if is_instance_valid(p) and p.health > 0:
			return true
	return false
