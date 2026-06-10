extends Node2D

var viewport_size: Vector2

@export var player             : PackedScene
@export var boss               : PackedScene
@export var player_spawn_point : Node2D
@export var boss_spawn_point   : Node2D
@export var random_spawns      : bool = true
@export var load_best_model    : bool = false
@export var is_new_train       : bool = false

# -------------------------------------------------------
# REWARDS — todo en la misma escala, acumulado equilibrado
# -------------------------------------------------------

# Terminales  (se suman una sola vez al final)
const REWARD_WIN              : float =  1.0
const REWARD_LOSE             : float = -1.0
const REWARD_FAST_WIN_BONUS   : float =   0.02   # por step restante al ganar
const REWARD_FAST_LOSE_BONUS  : float =  -0.01   # por step restante al perder rápido

# Por step (escala ~[-1, 1] por step para que en 600 steps → [-600, 600] simétrico)
const R_MOVING                : float =  0.05    # moverse
const R_STATIC                : float = -0.05    # quieto

const R_CLOSENESS_MAX         : float =  0.3     # máximo por proximidad
const R_TOO_FAR               : float = -0.1     # demasiado lejos

const R_AIM_MAX               : float =  0.3     # puntería perfecta
const R_DAMAGE_DEALT          : float =  0.5     # por cada punto de HP quitado / max_hp (normalizado)
const R_DAMAGE_TAKEN          : float = -0.2     # por cada punto de HP perdido / max_hp

const R_DODGE_MAX             : float =  0.1     # máximo por alejarse de bala (normalizado)

const MIN_PLAYER_DIST         : float = 400.0
const MIN_SPEED_THRESHOLD     : float = 20.0

# NN
var nn_client: NNClient

# Historial para deltas de salud
var last_boss_health  : float = 0.0
var last_player_health: float = 0.0
var last_dist_to_bullet: float = 0.0

var is_resetting: bool = false

func _ready() -> void:
	Engine.time_scale = 1.0
	_load_train_data()
	_init_nn_core()
	_reset_health_tracking()
	_spawn_entities()

func _physics_process(_delta: float) -> void:
	if is_resetting: return
	if not is_instance_valid(GlobalVars.boss) or GlobalVars.players.is_empty(): return
	
	GlobalVars.current_step += 1
	
	var current_inputs: Array = _get_inputs_for_nn()
	var epsilon: float = max(0.05, 0.3 - GlobalVars.current_episode * 0.0001)
	var reward: float  = _calculate_reward()
	GlobalVars.current_reward += reward
	
	var response = nn_client.request_action(current_inputs, reward, epsilon)
	
	GlobalVars.nn_outputs["move_dir"]   = response.get("move_dir",   [0.0, 0.0])
	GlobalVars.nn_outputs["shot_angle"] = response.get("shot_angle", 0.0)
	GlobalVars.nn_outputs["action"]     = response.get("action",     0)
	
	if _can_episode_end():
		_handle_episode_end()

# -------------------------------------------------------
# Rewards
# -------------------------------------------------------

func _calculate_reward() -> float:
	if not is_instance_valid(GlobalVars.boss) or GlobalVars.players.is_empty():
		return 0.0
	
	var p = GlobalVars.players[0]
	var b = GlobalVars.boss
	var reward = 0.0

	# 1. Daño infligido al jugador (positivo)
	var damage_dealt = last_player_health - p.health
	if damage_dealt > 0.0:
		# Normalizado entre 0 y 1 (si mata al jugador, da +1)
		reward += damage_dealt / p.max_health

	# 2. Daño recibido (negativo)
	var damage_taken = last_boss_health - b.health
	if damage_taken > 0.0:
		reward -= damage_taken / b.max_health

	# 3. Penalización por paso (incentiva ganar rápido)
	reward -= 0.01

	# 4. Bonus extra por matar al jugador (opcional, refuerza la victoria)
	if p.health <= 0.0:
		reward += 1.0   # para que el agente note que ganó

	# Actualizar historial para el próximo paso
	last_player_health = p.health
	last_boss_health = b.health

	return reward

func _calculate_final_reward() -> float:
	var steps_remaining = GlobalConst.MAX_STEP_FOR_EPISODE - GlobalVars.current_step
	if GlobalVars.players.is_empty():
		return REWARD_WIN  + steps_remaining * REWARD_FAST_WIN_BONUS
	elif is_instance_valid(GlobalVars.boss) and GlobalVars.boss.health <= 0.0:
		return REWARD_LOSE + steps_remaining * REWARD_FAST_LOSE_BONUS
	else:
		return REWARD_LOSE  # timeout = derrota

func _handle_episode_end() -> void:
	is_resetting = true
	
	var final_reward: float = _calculate_final_reward()
	nn_client.notify_episode_end(GlobalVars.current_episode, GlobalVars.current_reward, final_reward)
	_save_train_data()
	
	GlobalVars.episode_rewards.append(GlobalVars.current_reward)
	GlobalVars.current_episode += 1
	
	_reset_episode()
	_reset_health_tracking()

func _reset_health_tracking() -> void:
	await get_tree().physics_frame
	await get_tree().physics_frame
	
	if is_instance_valid(GlobalVars.boss):
		last_boss_health = GlobalVars.boss.health
	if not GlobalVars.players.is_empty() and is_instance_valid(GlobalVars.players[0]):
		last_player_health = GlobalVars.players[0].health
	last_dist_to_bullet = 0.0
	
	is_resetting = false

# -------------------------------------------------------
# Inputs — total debe coincidir con GlobalConst.INPUTS
# Boss: 35, Player: 5 → total = 40
# Asegurate de actualizar GlobalConst.INPUTS = 40
# -------------------------------------------------------

func _get_inputs_for_nn() -> Array:
	var inputs: Array = []
	if not is_instance_valid(GlobalVars.boss):
		for _i in range(GlobalConst.INPUTS): inputs.append(0.0)
		return inputs
	
	inputs.append_array(GlobalVars.boss.get_inputs())  # 35 floats

	if is_instance_valid(GlobalVars.boss.near_player):
		inputs.append_array(GlobalVars.boss.near_player.get_inputs())  # 5 floats
	else:
		for _i in range(5): inputs.append(0.0)

	# Garantiza exactamente GlobalConst.INPUTS elementos
	while inputs.size() < GlobalConst.INPUTS:
		inputs.append(0.0)
	if inputs.size() > GlobalConst.INPUTS:
		inputs = inputs.slice(0, GlobalConst.INPUTS)
	
	return inputs

# -------------------------------------------------------
# Utilidad
# -------------------------------------------------------

func _reset_episode() -> void:
	for bullet in GlobalVars.bullets:
		if is_instance_valid(bullet): bullet.queue_free()
	for p in GlobalVars.players:
		if is_instance_valid(p): p.queue_free()
	if is_instance_valid(GlobalVars.boss): GlobalVars.boss.queue_free()
	
	GlobalVars.boss          = null
	GlobalVars.bullets.clear()
	GlobalVars.players.clear()
	GlobalVars.shot_impact   = Vector2.ZERO
	GlobalVars.current_step  = 0
	GlobalVars.current_reward = 0.0
	GlobalVars.nn_outputs["move_dir"]   = [0.0, 0.0]
	GlobalVars.nn_outputs["shot_angle"] = 0.0
	GlobalVars.nn_outputs["action"]     = 0
	_spawn_entities()

func _init_nn_core() -> void:
	nn_client = NNClient.new()

func _spawn_entities() -> void:
	var player_instance = player.instantiate() as PlayerController
	var boss_instance   = boss.instantiate() as BossController
	viewport_size = get_viewport().get_visible_rect().size
	
	if not random_spawns:
		player_instance.global_position = player_spawn_point.global_position
		boss_instance.global_position   = boss_spawn_point.global_position
	else:
		var margin: float  = 200.0
		var max_attempts   = 20
		var player_pos     := Vector2.ZERO
		var boss_pos       := Vector2.ZERO
		for _i in range(max_attempts):
			player_pos = Vector2(randf_range(0.0, viewport_size.x), randf_range(0.0, viewport_size.y))
			boss_pos   = Vector2(randf_range(0.0, viewport_size.x), randf_range(0.0, viewport_size.y))
			if player_pos.distance_to(boss_pos) >= margin:
				break
		player_instance.global_position = player_pos
		boss_instance.global_position   = boss_pos

	get_tree().get_root().add_child.call_deferred(player_instance)
	get_tree().get_root().add_child.call_deferred(boss_instance)

func _can_episode_end() -> bool:
	if GlobalVars.players.is_empty(): return true
	if not is_instance_valid(GlobalVars.boss): return true
	if not is_instance_valid(GlobalVars.players[0]): return true
	return (
		GlobalVars.current_step >= GlobalConst.MAX_STEP_FOR_EPISODE
		or GlobalVars.boss.health   <= 0.0
		or GlobalVars.players[0].health <= 0.0
	)

func _load_train_data() -> void:
	if not is_new_train:
		var data = ExternalFileManager.read_json(GlobalConst.BEST_TRAIN_DATA_PATH)
		if data.is_empty():
			print("No hay datos para cargar: ", GlobalConst.BEST_TRAIN_DATA_PATH)
			return
		GlobalVars.current_episode   = data.get('episode', 0)
		GlobalVars.best_avg_reward   = data.get('best_avg_reward', -1e9)
		GlobalVars.best_avg_episode  = data.get('best_avg_episode', 0)
		GlobalVars.episode_rewards   = data.get('episodes_rewards', [])

func _save_train_data() -> void:
	var data: Dictionary = {
		'episode':        GlobalVars.current_episode,
		'best_avg_reward':  GlobalVars.best_avg_reward,
		'best_avg_episode': GlobalVars.best_avg_episode,
		'episodes_rewards': GlobalVars.episode_rewards
	}
	ExternalFileManager.save_data(data, GlobalConst.BEST_TRAIN_DATA_PATH)
