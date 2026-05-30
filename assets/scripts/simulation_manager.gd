extends Node2D

@export var players             : Array[PackedScene]
@export var spawn_player_points : Array[Node2D]
@export var boss_spawn          : Node2D
@export var boss                : PackedScene

# Constantes de normalización (deben coincidir con el viewport y los stats del boss/player)
const VIEWPORT_W      : float = 800.0
const VIEWPORT_H      : float = 320.0
const MAX_BOSS_HEALTH : float = 10000.0
const MAX_BOSS_DAMAGE : float = 200.0
const MAX_PLR_HEALTH  : float = 1000.0
const MAX_PHASES      : float = 4.0

# ─────────────────────────────────────────
#  Sistema de recompensas
# ─────────────────────────────────────────
const REWARD_DAMAGE_DEALT    : float =  0.5   # Por cada punto de daño hecho al jugador
const REWARD_SURVIVE_STEP    : float =  0.01  # Por sobrevivir un step
const REWARD_WIN_EPISODE     : float =  50.0  # El boss mata al jugador
const REWARD_LOSE_EPISODE    : float = -50.0  # El boss muere
const REWARD_APPROACH_PLAYER : float =  0.005 # Por acercarse al jugador

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
var prev_boss_dist     : float = 0.0

func _ready() -> void:
	_setup_nn()
	spawn_entities()

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
func spawn_entities() -> void:
	# Limpia instancias anteriores
	for p in player_instances:
		if is_instance_valid(p):
			p.queue_free()
	player_instances.clear()
	GlobalVars.players.clear()
	
	if is_instance_valid(boss_instance):
		boss_instance.queue_free()
	boss_instance = null
	
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
	episode_running    = true

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
	
	# Actualiza estado previo para el próximo step
	prev_player_health = _get_total_player_health()
	prev_boss_dist     = _get_dist_boss_to_nearest_player()

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
	spawn_entities()

# ─────────────────────────────────────────
#  Construcción del vector de inputs (normalizado)
# ─────────────────────────────────────────
func _build_input_vec() -> Array:
	if not is_instance_valid(boss_instance):
		return _zero_input_vec()
	
	var b  : BossController  = boss_instance
	var np : PlayerController = b.near_player  # puede ser null
	
	var input_vec : Array = [
		b.health            / MAX_BOSS_HEALTH,
		b.total_damage      / MAX_BOSS_DAMAGE,
		b.global_position.x / VIEWPORT_W,
		b.global_position.y / VIEWPORT_H,
		b.move_dir.x,                          # ya en [-1, 1]
		b.move_dir.y,                          # ya en [-1, 1]
		float(b.boss_pashe) / MAX_PHASES,
	]
	
	# Stats del jugador más cercano
	if np:
		input_vec.append_array([
			np.health            / MAX_PLR_HEALTH,
			np.global_position.x / VIEWPORT_W,
			np.global_position.y / VIEWPORT_H,
			np.global_position.x / VIEWPORT_W,  # near_player pos (redundante si es el mismo, útil con multi-player)
			np.global_position.y / VIEWPORT_H,
		])
	else:
		input_vec.append_array([0.0, 0.0, 0.0, 0.0, 0.0])
	
	return input_vec  # 12 valores

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
	
	# Recompensa por dañar al jugador este step
	var current_player_health : float = _get_total_player_health()
	var damage_dealt          : float = prev_player_health - current_player_health
	if damage_dealt > 0.0:
		reward += damage_dealt * REWARD_DAMAGE_DEALT
	
	# Recompensa por acercarse al jugador
	var current_dist : float = _get_dist_boss_to_nearest_player()
	if current_dist < prev_boss_dist:
		reward += REWARD_APPROACH_PLAYER
	
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
