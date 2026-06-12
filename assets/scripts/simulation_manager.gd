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
# --- Recompensas terminales ---
const TERM_WIN_BASE          : float = 50.0   # Victoria: matar al jugador
const TERM_WIN_BONUS_MAX     : float = 10.0   # Bono adicional por tiempo restante (multiplicado por fracción de tiempo)
const TERM_LOSE              : float = -10.0  # Derrota: boss muere
const TERM_TIMEOUT_MAX_PEN   : float = -5.0   # Penalización base por timeout (luego se multiplica por salud restante del jugador)

# --- Recompensas por paso (eventos) ---
const R_DAMAGE_DEALT_MAX     : float = 5.0    # Por dañar al jugador
const R_DAMAGE_TAKEN_MAX     : float = -20.0  # Por perder vida
const R_PROXIMITY_MAX        : float = 0.005  # Por estar pegado al jugador
const R_CLOSING_DIST_MAX     : float = 0.01   # Por acercarse al jugador
const R_GOOD_AIM             : float = 0.05   # Por apuntar bien
const R_NEAR_BULLET          : float = 3.0    # Por la bala pasar cerca del jugador

# NN
var nn_client: NNClient

# Historial para deltas de salud
var last_boss_health  : float = 0.0
var last_player_health: float = 0.0
var last_dist_to_bullet: float = 0.0
var last_dist_to_player: float = 0.0
var last_shot_impact: Vector2 = Vector2.ZERO
var is_resetting: bool = false

func _ready() -> void:
	Engine.time_scale = 1.0
	_load_train_data()
	_init_nn_core()
	_reset_health_tracking()
	_reset_episode()

func _physics_process(_delta: float) -> void:
	if is_resetting: return
	if not is_instance_valid(GlobalVars.boss) or GlobalVars.players.is_empty(): return
	
	GlobalVars.current_step += 1
	
	var current_inputs: Array = _get_inputs_for_nn()
	var reward: float  = _calculate_reward()
	GlobalVars.current_reward += reward
	
	var response = nn_client.request_action(current_inputs, reward)
	
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
	
	# --- Daño infligido al jugador ---
	var damage_dealt = last_player_health - p.health
	if damage_dealt > 0.0:
		var dmg_ratio = clamp(damage_dealt / p.max_health, 0.0, 1.0)
		reward += dmg_ratio * R_DAMAGE_DEALT_MAX
	
	# --- Daño recibido (penalización) ---
	var damage_taken = last_boss_health - b.health
	if damage_taken > 0.0:
		var taken_ratio = clamp(damage_taken / b.max_health, 0.0, 1.0)
		reward += taken_ratio * R_DAMAGE_TAKEN_MAX   # R_DAMAGE_TAKEN_MAX es negativo
	
	# --- Proximidad al jugador (recompensa por estar cerca) ---
	var dist = b.global_position.distance_to(p.global_position)
	var max_dist = viewport_size.length()
	var dist_norm = clamp(dist / max_dist, 0.0, 1.0)
	reward += (1.0 - dist_norm) * R_PROXIMITY_MAX
	
	# --- Cambio en la distancia (acercarse es bueno) ---
	if last_dist_to_player > 0.0:
		var dist_change = last_dist_to_player - dist
		var delta_norm = clamp(dist_change / max_dist, -0.1, 0.1)   # limita cambio por paso
		reward += delta_norm * R_CLOSING_DIST_MAX
	last_dist_to_player = dist
	
	# --- Incentivo por atacar (acción 1) ---
	if b.current_action == 1:
		var steps_since_shot = GlobalVars.current_step - b.shot_attack.last_shot_step
		if steps_since_shot < 30:  # disparó en los últimos ~0.5s
			var angle_diff = abs(wrapf(
				atan2(p.global_position.y - b.global_position.y,
					  p.global_position.x - b.global_position.x) - b.shot_angle,-PI, PI))
			var aim_reward = (1.0 - angle_diff / PI) * R_GOOD_AIM  # reducido a 0.05
			reward += aim_reward
	
	# --- Si la bala pasa cerca del jugador ---
	if GlobalVars.shot_impact != last_shot_impact:
		var dist_to_player = GlobalVars.shot_impact.distance_to(p.global_position)
		if dist_to_player < 100.0:
			reward += R_NEAR_BULLET  # La bala del boss llegó cerca del jugador
		last_shot_impact = GlobalVars.shot_impact
	
	# Actualizar valores para el próximo paso
	last_player_health = p.health
	last_boss_health   = b.health
	return reward

func _calculate_final_reward() -> float:
	var steps_remaining = GlobalConst.MAX_STEP_FOR_EPISODE - GlobalVars.current_step
	var time_bonus = steps_remaining / float(GlobalConst.MAX_STEP_FOR_EPISODE)  # 0..1
	
	# Victoria: jugador muerto
	if GlobalVars.players.is_empty() or (GlobalVars.players[0].health <= 0.0):
		return TERM_WIN_BASE + time_bonus * TERM_WIN_BONUS_MAX
	
	# Derrota: boss muerto
	elif is_instance_valid(GlobalVars.boss) and GlobalVars.boss.health <= 0.0:
		return TERM_LOSE
	
	# Timeout: se acabó el tiempo sin muerte
	else:
		var player_hp_ratio = GlobalVars.players[0].health / GlobalVars.players[0].max_health
		# Penalización proporcional a la salud que le queda al jugador (menos salud = menos penalización)
		return TERM_TIMEOUT_MAX_PEN * (1.0 - player_hp_ratio)

func _handle_episode_end() -> void:
	is_resetting = true
	
	var final_reward: float = _calculate_final_reward()
	nn_client.notify_episode_end(GlobalVars.current_episode, GlobalVars.current_reward, final_reward)
	
	# Guarda la mejor recompensa
	if GlobalVars.current_reward > GlobalVars.best_avg_reward:
		GlobalVars.best_avg_reward = GlobalVars.current_reward
		GlobalVars.best_avg_episode = GlobalVars.current_episode
	_save_train_data()
	
	GlobalVars.episode_rewards.append(GlobalVars.current_reward)
	GlobalVars.current_episode += 1
	
	_reset_episode()
	_reset_health_tracking()

func _reset_health_tracking() -> void:
	# Espera hasta que boss y player estén efectivamente en el árbol
	var timeout = 0
	while (not is_instance_valid(GlobalVars.boss) or GlobalVars.players.is_empty()) and timeout < 60:
		await get_tree().physics_frame
		timeout += 1
	if not GlobalVars.players.is_empty() and is_instance_valid(GlobalVars.players[0]) and is_instance_valid(GlobalVars.boss):
		last_dist_to_player = GlobalVars.boss.global_position.distance_to(GlobalVars.players[0].global_position)
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
	GlobalVars.shot_impact   = Vector2(viewport_size.x / 2, viewport_size.y / 2)
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
		var margin_spawn: float = 80.0
		var margin: float  = 200.0
		var max_attempts   = 20
		var player_pos     := Vector2.ZERO
		var boss_pos       := Vector2.ZERO
		for _i in range(max_attempts):
			player_pos = Vector2(
				randf_range(margin_spawn, viewport_size.x - margin_spawn),
				randf_range(margin_spawn, viewport_size.y - margin_spawn)
			)
			boss_pos   = Vector2(
				randf_range(margin_spawn, viewport_size.x - margin_spawn),
				randf_range(margin_spawn, viewport_size.y - margin_spawn)
			)
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
