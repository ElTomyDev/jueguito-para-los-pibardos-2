extends Node2D

@export var players             : Array[PackedScene]
@export var spawn_player_points : Array[Node2D]
@export var boss_spawn          : Node2D
@export var boss                : PackedScene

# ─────────────────────────────────────────
#  Sistema de recompensas
# ─────────────────────────────────────────
const REWARD_DAMAGE_DEALT    : float =  0.01   # Por cada punto de daño hecho al jugador
const REWARD_SURVIVE_STEP    : float =  0.01  # Por sobrevivir un step
const REWARD_WIN_EPISODE     : float =  100.0  # El boss mata al jugador
const REWARD_LOSE_EPISODE    : float = -30.0  # El boss muere
const REWARD_APPROACH_PLAYER : float =  0.005 # Por acercarse al jugador
const REWARD_FOR_SURVIVE     : float = 0.05  # Por sobrevivir 
const REWARD_DODGE_BULLET    : float = 0.05 # Por esquivar balas
const REWARD_DAMAGE_RECIBE   : float = 0.03    # Por recibir daño

# ─────────────────────────────────────────
#  Nodos de la NN (se instancian en _ready)
# ─────────────────────────────────────────
var nn          : NeuralNetwork
var trainer     : NNTrainer
var persistence : NNPersistence

# Referencias a entidades vivas
var boss_instance   : BossController   = null
var player_instances: Array            = []

# Estado del episodio
var episode_running    : bool  = false
var prev_player_health : float = 0.0
var prev_boss_health : float = 0.0
var prev_boss_dist     : float = 0.0

# Contador de frames para debug
var _debug_frame_count : int = 0 

func _ready() -> void:
	_setup_nn()
	start_simulation()


# ─────────────────────────────────────────
#  Inicializa y carga la red neuronal
# ─────────────────────────────────────────
func _setup_nn() -> void:
	nn          = NeuralNetwork.new()
	trainer     = NNTrainer.new()
	persistence = NNPersistence.new()
	
	add_child(nn)
	add_child(trainer)
	add_child(persistence)
	
	persistence.load_into(nn)  # Carga pesos previos si existen
	trainer.setup(nn)

# ─────────────────────────────────────────
#  Spawn de entidades
# ─────────────────────────────────────────
func start_simulation() -> void:
	_debug_frame_count = 0
	clean_entities()
	
	# Spawnea jugadores
	for idx in range(players.size()):
		var player_instance = players[idx].instantiate()
		player_instance.global_position = spawn_player_points[idx].global_position
		add_child(player_instance)
		player_instances.append(player_instance)
	
	# Spawnea boss
	boss_instance = boss.instantiate()
	boss_instance.global_position = boss_spawn.global_position
	add_child(boss_instance)
	
	# Estado inicial del episodio
	prev_player_health = _get_total_player_health()
	prev_boss_dist     = _get_dist_boss_to_nearest_player()
	prev_boss_health = boss_instance.health
	episode_running    = true

func clean_entities() -> void:
	# Limpia instancias anteriores
	for p in player_instances:
		if is_instance_valid(p):
			p.queue_free()
	player_instances.clear()
	GlobalVars.players.clear()
	
	if is_instance_valid(boss_instance):
		boss_instance.queue_free()
	boss_instance = null

# ─────────────────────────────────────────
#  Loop principal: corre cada physics frame
# ─────────────────────────────────────────
func _physics_process(_delta: float) -> void:
	if not episode_running:
		return
	
	_check_episode_end()
	
	if not episode_running:
		return
	
	if not is_instance_valid(boss_instance):
		return
	
	# Construye el vector de inputs normalizado
	var input_vec : Array = _build_input_vec()
	
	# Calcula recompensa del step actual
	var reward : float = _compute_step_reward()
	
	# Hace forward + guarda experiencia
	var output : Array = trainer.step(input_vec, reward)
	
	# Actualiza GlobalVars con el output de la red
	_apply_nn_output(output)
	
	# Debug: imprime inputs y outputs los primeros 3 frames de cada episodio
	_debug_frame_count += 1
	if _debug_frame_count <= 3:
		pass
		#print("=== FRAME ", _debug_frame_count, " ===")
		#print("INPUT VEC (", input_vec.size(), " valores): ", input_vec)
		#print("OUTPUT raw: ", output)
		#print("  move_dir  → ", GlobalVars.nn_outputs['move_dir'])
		#print("  shot_dir  → ", GlobalVars.nn_outputs['shot_dir'])
		#print("  action    → ", GlobalVars.nn_outputs['current_action'])
		#print("  reward    → ", reward)
	
	# Actualiza estado previo para el próximo step
	prev_player_health = _get_total_player_health()
	prev_boss_dist     = _get_dist_boss_to_nearest_player()
	prev_boss_health = boss_instance.health if is_instance_valid(boss_instance) else 0.0
# ─────────────────────────────────────────
#  Chequea si el episodio terminó
# ─────────────────────────────────────────
func _check_episode_end() -> void:
	var boss_alive    : bool = is_instance_valid(boss_instance) and boss_instance.health > 0
	var players_alive : bool = _any_player_alive()
	
	if not boss_alive:
		_end_episode(false)  # Boss murió → perdió
	elif not players_alive:
		_end_episode(true)   # Jugadores muertos → ganó

# ─────────────────────────────────────────
#  Finaliza el episodio, entrena y respawnea
# ─────────────────────────────────────────
func _end_episode(boss_won: bool) -> void:
	episode_running = false
	
	var final_reward : float = REWARD_WIN_EPISODE if boss_won else REWARD_LOSE_EPISODE
	trainer.end_episode(final_reward)
	persistence.save(nn)
	
	print("Episodio ", trainer.episode_count, 
		  " | Boss ganó: ", boss_won, 
		  " | Recompensa total: ", trainer.total_reward_last_episode)
	
	# Pequeña pausa antes del respawn (opcional, podés sacarla)
	await get_tree().create_timer(1.5).timeout
	start_simulation()

# ─────────────────────────────────────────
#  Construcción del vector de inputs (normalizado)
# ─────────────────────────────────────────
func _build_input_vec() -> Array:
	if not is_instance_valid(boss_instance):
		return _zero_input_vec()
	
	var b  : BossController  = boss_instance
	var np : PlayerController = b.near_player  # puede ser null
	
	var input_vec : Array = b.stats_normalized()
	
	# Stats del jugador más cercano
	if np:
		input_vec.append_array(np.stats_normalized())
	else:
		input_vec.append_array([0.0, 0.0, 0.0])
	
	return input_vec 

func _zero_input_vec() -> Array:
	var v : Array = []
	for _i in range(12):
		v.append(0.0)
	return v

# ─────────────────────────────────────────
#  Cálculo de recompensa por step
# ─────────────────────────────────────────
func _compute_step_reward() -> float:
	var reward : float = REWARD_SURVIVE_STEP
	
	# Recompensa por estar vivo
	if prev_boss_health > 0.0:
		reward += REWARD_FOR_SURVIVE
	
	# Recompensa por dañar al jugador este step
	var current_player_health : float = _get_total_player_health()
	# Daño hecho: pequeño por step, grande por ganar
	var damage_dealt : float = prev_player_health - current_player_health
	if damage_dealt > 0.0:
		reward += damage_dealt * REWARD_DAMAGE_DEALT
	
	# Recibir daño: penalización moderada
	var damage_received : float = prev_boss_health - boss_instance.health
	if damage_received > 0.0:
		reward -= damage_received * REWARD_DAMAGE_RECIBE  # 200 × 0.05 = 10.0
	elif is_instance_valid(boss_instance.near_bullet): # Esquivar activamente
		reward += REWARD_DODGE_BULLET
	return reward

# ─────────────────────────────────────────
#  Aplica el output de la red a GlobalVars
#  para que boss_controller lo lea en update_boss()
# ─────────────────────────────────────────
func _apply_nn_output(output: Array) -> void:
	GlobalVars.nn_outputs = {
		"move_dir"      : [output[0], output[1]],
		"shot_dir"      : [output[2], output[3]],
		"current_action": 1 if output[4] >= 0.5 else 0,
	}

# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────
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

func _get_dist_boss_to_nearest_player() -> float:
	if not is_instance_valid(boss_instance):
		return 0.0
	if not is_instance_valid(boss_instance.near_player):
		return INF
	return boss_instance.global_position.distance_to(boss_instance.near_player.global_position)
