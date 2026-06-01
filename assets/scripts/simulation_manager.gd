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
const REWARD_NEAR_BULLET     : float = 0.005   # Por disparar cerca del jugador

const PROXIMITY_MAX_RANGE    : float = 60.0    # Rango de recompensa para proximidad de bala
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

var total_reward_step: float = 0.0

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
	if not is_instance_valid(boss_instance): return
	
	var state = _get_neural_network_inputs()
	var output = nn.forward(state)
	_apply_actions(output)
	
	boss_instance.move_and_slide()
	
	# 4. Calcular recompensa basada en el cambio de estado
	var next_state = _get_neural_network_inputs() 
	var reward = _calculate_reward()
	var done = false
	if (current_step >= MAX_STEPS_PER_EPISODE) or (boss_instance.health <= 0.0 or GlobalVars.players[0].health <= 0.0):
		print("DEBUG: Epísodio terminado. Salud jefe: ", boss_instance.health)
		done = true
	
	# 6. Entrenamiento en línea
	trainer.train_actor_critic(state, output.slice(0,4), reward, next_state, done)
	
	current_step += 1
	if done: _reset_simulation()

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
	print("Finalizando Episodio ", trainer.episode_count, " - Pasos: ", current_step, " - Sigma: ", trainer.current_sigma, " - Total Reward: ", total_reward_step + final_reward)
	trainer.end_episode()
	# Guardar progreso periódicamente
	if trainer.episode_count % 10 == 0:
		persistence.save(nn, trainer)
		
	total_reward_step = 0.0
	_reset_simulation()

# ─────────────────────────────────────────
#  Mapeo de Variables de Entrada (15 Inputs)
# ─────────────────────────────────────────
func _get_neural_network_inputs() -> Array:
	var inputs: Array = []
	if not is_instance_valid(boss_instance):
		for i in range(15): inputs.append(0.0)
		return inputs
	
	# Boss: Pos (0 a 1) y Vel (normalizada por max_speed)
	inputs.append_array([
		boss_instance.global_position.x / viewport_size.x,
		boss_instance.global_position.y / viewport_size.y,
		clampf(boss_instance.velocity.x / boss_instance.max_speed, -1.0, 1.0),
		clampf(boss_instance.velocity.y / boss_instance.max_speed, -1.0, 1.0),
		boss_instance.health / boss_instance.initial_health
	])
	
	# Jugador: Relativo (normalizado por 1000px)
	var near_player = boss_instance.near_player
	if is_instance_valid(near_player):
		var dir = (near_player.global_position - boss_instance.global_position) / 1000.0
		inputs.append_array([clampf(dir.x, -1.0, 1.0), clampf(dir.y, -1.0, 1.0), 0.0, 0.0, 0.0]) # Ajustar según sensores
	else:
		inputs.append_array([0.0, 0.0, 0.0, 0.0, 0.0])
		
	# Bala: Relativo (normalizado por 500px)
	var near_bullet = boss_instance.near_bullet
	if is_instance_valid(near_bullet):
		var dir = (near_bullet.global_position - boss_instance.global_position) / 500.0
		inputs.append_array([clampf(dir.x, -1.0, 1.0), clampf(dir.y, -1.0, 1.0), 0.0, 0.0, 0.0])
	else:
		inputs.append_array([1.0, 1.0, 0.0, 0.0, 0.0])
		
	return inputs

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
	# 1. Calcular recompensa por proximidad de balas al jugador
	var bullets = get_tree().get_nodes_in_group("Bullets")
	var player = GlobalVars.players[0] if GlobalVars.players.size() > 0 else null
	
	if player and bullets.size() > 0:
		for b in bullets:
			# Solo evaluamos balas que no sean del Boss (o las que quieras premiar)
			if is_instance_valid(b) and b.from_group != "Players": 
				var dist = b.global_position.distance_to(player.global_position)
				if dist <= PROXIMITY_MAX_RANGE:
					reward += REWARD_NEAR_BULLET * (1.0 - (dist / PROXIMITY_MAX_RANGE))
					
	total_reward_step += reward
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

func _get_game_state() -> Array:
	var state = []
	if not is_instance_valid(boss_instance): return []
	
	# 1-2. Posición relativa al jugador más cercano
	var near_player = boss_instance._get_near_player()
	var p_pos = near_player.global_position if near_player else Vector2.ZERO
	var rel_pos = ((p_pos - boss_instance.global_position) / 1000.0).clamp(Vector2(-1, -1), Vector2(1, 1))
	state.append_array([rel_pos.x, rel_pos.y])
	
	# 3-4. Distancia y dirección a la bala más cercana
	var near_bullet = boss_instance.near_bullet
	if near_bullet:
		var b_rel = ((near_bullet.global_position - boss_instance.global_position) / 500.0).clamp(Vector2(-1, -1), Vector2(1, 1))
		state.append_array([b_rel.x, b_rel.y])
	else:
		state.append_array([1.0, 1.0]) # No hay peligro
		
	# 5. Vida del boss (0.0 a 1.0)
	state.append(boss_instance.health / boss_instance.initial_health)
	
	# 6-7. Velocidad actual del boss
	state.append_array([boss_instance.velocity.x / 300.0, boss_instance.velocity.y / 300.0])
	
	# 8-15. Rellenar con datos adicionales (limites, ángulo, cooldown, etc.)
	# Es vital que siempre devuelva 15 floats exactos para tu capa de entrada.
	while state.size() < 15:
		state.append(0.0)
		
	return state

func _apply_actions(output: Array) -> Array:
	# output: [mov_x, mov_y, angulo, disparo, valor_critic]
	# Usamos los primeros 4 para controlar al boss
	if not is_instance_valid(boss_instance): return [0.0, 0.0, 0.0, 0.0]
	
	var shot_angle : float = output[2] * PI
	if GlobalVars.nn_outputs.is_empty():
		GlobalVars.nn_outputs = {
			"move_dir"      : [output[0], output[1]],
			"shot_angle"    : shot_angle,
			"current_action": force_attack if force_attack_mode else int(output[3]),
		}
	else:
		GlobalVars.nn_outputs["move_dir"] = Vector2(output[0], output[1])
		GlobalVars.nn_outputs["shot_angle"] = shot_angle
		GlobalVars.nn_outputs["current_action"] = 1 if output[3] > 0.5 else 0
	
	# Retornamos la acción aplicada para que el trainer sepa qué se ejecutó
	return [output[0], output[1], output[2], output[3]]

func _get_game_next_state() -> Array:
	# Se llama tras un frame de física para ver cómo cambió el entorno
	return _get_game_state()
