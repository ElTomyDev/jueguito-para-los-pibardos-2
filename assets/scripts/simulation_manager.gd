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
# --- Terminales ---
const TERM_WIN_BASE        : float = 100.0
const TERM_WIN_TIME_BONUS  : float = 30.0
const TERM_LOSE            : float = -100.0
const TERM_TIMEOUT_PEN     : float = -30.0

# --- Supervivencia ---
const R_DAMAGE_TAKEN_MAX   : float = -120.0
const R_DODGE_DISTANCE_MIN : float = 60.0
const R_DODGE_DISTANCE_MAX : float = 500.0
const R_ACTIVE_DODGE_GAIN   : float = 0.4   # por cada unidad de distancia que se aleja
const R_ACTIVE_DODGE_MAX    : float = 10.0  # máx por bala por frame (evita explotar)
const R_PASSIVE_DODGE       : float = 1.0   # mantener una pequeña recompensa si la bala desaparece (por si acaso)

# --- Precisión ---
const R_SHOT_GOOD_AIM        : float = 45.0
const R_SHOT_HIT_PLAYER      : float = 50.5
const R_DAMAGE_DEALT_MAX     : float = 95.0
const R_GOOD_AIM             : float = 1.0
const R_SHOT_AND_NEAR_PLAYER : float = 0.3

# --- Inactividad y movimiento ---
const R_IDLE_PENALTY        : float = -0.15
const IDLE_STREAK_THRESHOLD : int   = 20
const R_NEAR_WALLS          : float = -0.2
const R_STATIC              : float = -0.1

# --- Umbrales y margenes ---
const WALL_MARGIN      : float = 85.0
const MIN_VEL_TRESHOLD : float = 5.0

# NN
var nn_client: NNClient

# Historial para deltas de salud
var last_boss_health  : float = 0.0
var last_player_health: float = 0.0
var last_shot_impact: Vector2 = Vector2.ZERO

# Tracking de aim al momento del disparo
var last_boss_shot_step    : int = -1
var _idle_streak           : int = 0       # pasos consecutivos en action=0
# Tracking de balas del jugador para reward de esquive
var _player_bullets_near_last_frame: Dictionary = {}  # bullet_id → distancia anterior

var is_resetting: bool = false

func _ready() -> void:
	Engine.time_scale = 1.0
	_load_train_data()
	_init_nn_core()
	_reset_episode()

func _process(delta: float) -> void:
	if not GlobalVars.players.is_empty() and is_instance_valid(GlobalVars.players[0]):
		if Input.is_action_just_pressed("is_automatic_player"):
			GlobalVars.players[0].is_automatic = !GlobalVars.players[0].is_automatic

func _physics_process(_delta: float) -> void:
	if is_resetting: return
	if not is_instance_valid(GlobalVars.boss) or GlobalVars.players.is_empty(): return
	
	nn_client.poll()
	
	GlobalVars.current_step += 1
	
	var current_inputs: Array = _get_inputs_for_nn()
	var reward: float  = _calculate_reward()
	GlobalVars.current_reward += reward
	
	# Solo enviar si no hay una petición activa
	if not nn_client.is_busy():
		nn_client.request_action(current_inputs, reward)
	
	var response = nn_client.get_last_action()
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
	
	# ── 1. SUPERVIVENCIA: daño recibido ─────────────────────────────────────
	var damage_taken = last_boss_health - b.health
	if damage_taken > 0.0:
		var taken_ratio = clamp(damage_taken / b.max_health, 0.0, 1.0)
		reward += taken_ratio * R_DAMAGE_TAKEN_MAX  # negativo
	
	# ── 1b. SUPERVIVENCIA: esquivar balas del jugador ────────────────────────
	# Primero actualizamos el frame actual con dist + hit_target
	var current_player_bullets: Dictionary = {}
	for bullet in GlobalVars.bullets:
		if is_instance_valid(bullet) and bullet.from_group == "Players":
			var dist = b.global_position.distance_to(bullet.global_position)
			current_player_bullets[bullet.get_instance_id()] = {
				"dist": dist,
				"hit": bullet.hit_target
			}
	
	for bullet_id in _player_bullets_near_last_frame:
		var prev = _player_bullets_near_last_frame[bullet_id]
		var prev_dist = prev["dist"]
		var prev_hit = prev["hit"]
		# Si la bala aún existe en este frame
		if current_player_bullets.has(bullet_id):
			var curr_dist = current_player_bullets[bullet_id]["dist"]
			var curr_hit = current_player_bullets[bullet_id]["hit"]
			# Si no ha impactado todavía
			if not curr_hit and not prev_hit and curr_dist < R_DODGE_DISTANCE_MAX:
				var dist_change = prev_dist - curr_dist   # positiva si se acerca, negativa si se aleja
				if dist_change < 0:  # se está alejando
					var danger_factor = clamp(1.0 - (curr_dist / R_DODGE_DISTANCE_MAX), 0.0, 1.0)
					var active_reward = clamp(abs(dist_change) * R_ACTIVE_DODGE_GAIN * (1.0 + danger_factor), 0.0, R_ACTIVE_DODGE_MAX)
					reward += active_reward
		else:
			# La bala desapareció este frame: recompensa pasiva (opcional, pero baja)
			if not prev_hit and prev_dist < R_DODGE_DISTANCE_MAX and prev_dist > R_DODGE_DISTANCE_MIN:
				reward += R_PASSIVE_DODGE
	_player_bullets_near_last_frame = current_player_bullets
	
	# ── 2. PRECISIÓN: aim en el momento del disparo ──────────────────────────
	var boss_shot_step = b.shot_attack.last_shot_step
	if boss_shot_step > last_boss_shot_step and boss_shot_step > 0:
		last_boss_shot_step = boss_shot_step
		if is_instance_valid(b.near_player):
			var angle_to_player = atan2(
				p.global_position.y - b.global_position.y,
				p.global_position.x - b.global_position.x
			)
			var angle_diff = abs(wrapf(b.shot_angle - angle_to_player, -PI, PI))
			var aim_quality = clamp(1.0 - (angle_diff / (PI / 2.0)), 0.0, 1.0)
			reward += aim_quality * R_SHOT_GOOD_AIM
	
	# ── 2b. PRECISIÓN: bala del boss que impactó al jugador ──────────────────
	if GlobalVars.shot_impact != last_shot_impact:
		var dist_impact_to_player = GlobalVars.shot_impact.distance_to(p.global_position)
		if dist_impact_to_player < 80.0:
			reward += R_SHOT_HIT_PLAYER
		last_shot_impact = GlobalVars.shot_impact
	
	# ── 2c. PRECISIÓN: Daño infligido al jugador ────────────────────────────────────────
	var damage_dealt = last_player_health - p.health
	if damage_dealt > 0.0:
		var dmg_ratio = clamp(damage_dealt / p.max_health, 0.0, 1.0)
		reward += dmg_ratio * R_DAMAGE_DEALT_MAX
	
	# ── 2d. PRECISIÓN: recompensa por frame cuando está en modo ataque ──────────
	if is_instance_valid(b.near_player) and b.current_action == 1:
		var angle_to_player_dense = atan2(
			p.global_position.y - b.global_position.y,
			p.global_position.x - b.global_position.x
		)
		var angle_diff_dense = abs(wrapf(b.shot_angle - angle_to_player_dense, -PI, PI))
		var aim_quality_dense = clamp(1.0 - (angle_diff_dense / (PI / 2.0)), 0.0, 1.0)
		reward += aim_quality_dense * R_GOOD_AIM
	
	# ── 2e. PRECISIÓN: recompensa por por acercarse al jugador en modo ataque ──────────
	if b.current_action == 1 and is_instance_valid(b.near_player):
		var dist = b.global_position.distance_to(b.near_player.global_position)
		var proximity_bonus = clamp(1.0 - (dist / viewport_size.length()), 0.0, 1.0)
		reward += proximity_bonus * R_SHOT_AND_NEAR_PLAYER  # pequeño, por frame
	
	# ── Penalización por inactividad prolongada ──────────────────────────────
	if b.current_action == 0:
		_idle_streak += 1
		if _idle_streak > IDLE_STREAK_THRESHOLD:
			reward += R_IDLE_PENALTY * (_idle_streak - IDLE_STREAK_THRESHOLD)
	else:
		_idle_streak = 0
	
	# ── Penalización por estar cerca de las paredes ──────────────────────────────
	var too_close_to_wall = (
		b.global_position.x < WALL_MARGIN or
		b.global_position.x > viewport_size.x - WALL_MARGIN or
		b.global_position.y < WALL_MARGIN or
		b.global_position.y > viewport_size.y - WALL_MARGIN
	)
	if too_close_to_wall:
		reward += R_NEAR_WALLS
	
	# ── Penalización por quedarse quieto ──────────────────────────────
	if b.velocity.length() < MIN_VEL_TRESHOLD:
		reward += R_STATIC
	
	# ── Actualizar tracking ──────────────────────────────────────────────────
	last_player_health = p.health
	last_boss_health   = b.health
	
	return reward

func _calculate_final_reward() -> float:
	var steps_used = float(GlobalVars.current_step)
	var max_steps  = float(GlobalConst.MAX_STEP_FOR_EPISODE)
	var time_ratio = 1.0 - clamp(steps_used / max_steps, 0.0, 1.0)
	
	if GlobalVars.players.is_empty() or \
	   (not GlobalVars.players.is_empty() and GlobalVars.players[0].health <= 0.0):
		return TERM_WIN_BASE + time_ratio * TERM_WIN_TIME_BONUS  # 100..130
	
	if is_instance_valid(GlobalVars.boss) and GlobalVars.boss.health <= 0.0:
		var player_dmg_ratio = 0.0
		if not GlobalVars.players.is_empty() and is_instance_valid(GlobalVars.players[0]):
			player_dmg_ratio = 1.0 - (GlobalVars.players[0].health / GlobalVars.players[0].max_health)
		return TERM_LOSE + player_dmg_ratio * 40.0  # entre -100 y -60
	
	var player_hp_ratio = 1.0
	if not GlobalVars.players.is_empty() and is_instance_valid(GlobalVars.players[0]):
		player_hp_ratio = GlobalVars.players[0].health / GlobalVars.players[0].max_health
	return TERM_TIMEOUT_PEN * player_hp_ratio  # entre -30 y 0

func _handle_episode_end() -> void:
	is_resetting = true
	
	var final_reward: float = _calculate_final_reward()
	nn_client.notify_episode_end(GlobalVars.current_episode, GlobalVars.current_reward, final_reward)
	
	# Guarda la mejor recompensa
	if GlobalVars.current_reward > GlobalVars.best_avg_reward:
		GlobalVars.best_avg_reward = GlobalVars.current_reward
		GlobalVars.best_avg_episode = GlobalVars.current_episode
	_save_train_data()
	
	GlobalVars.current_episode += 1
	GlobalVars.episode_rewards.append(GlobalVars.current_reward)
	
	_reset_episode()

# ----------
# Inputs
# ----------
func _get_inputs_for_nn() -> Array:
	var inputs: Array = []
	if not is_instance_valid(GlobalVars.boss): # Crea 0.0 por la cantidad de inputs que devuelve SOLO el boss
		for _i in range(32): inputs.append(0.0)
		return inputs
	
	inputs.append_array(GlobalVars.boss.get_inputs())
	
	if is_instance_valid(GlobalVars.boss.near_player):
		inputs.append_array(GlobalVars.boss.near_player.get_inputs())
	else:# Crea 0.0 por la cantidad de inputs que devuelve SOLO el jugador
		for _i in range(5): inputs.append(0.0)
	
	assert(inputs.size() == GlobalConst.INPUTS,
		"_get_inputs_for_nn() retornó %d floats, se esperaban %d" % [inputs.size(), GlobalConst.INPUTS])
	
	return inputs

# ------------
# Utilidad
# ------------
func _reset_episode() -> void:
	for bullet in GlobalVars.bullets:
		if is_instance_valid(bullet): bullet.queue_free()
	for p in GlobalVars.players:
		if is_instance_valid(p): p.queue_free()
	if is_instance_valid(GlobalVars.boss): GlobalVars.boss.queue_free()
	
	last_boss_shot_step = -1
	_idle_streak = 0
	_player_bullets_near_last_frame = {}
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
	
	boss_instance.boss_ready.connect(_on_boss_ready.bind(player_instance, boss_instance), CONNECT_ONE_SHOT)

	get_tree().get_root().add_child.call_deferred(player_instance)
	get_tree().get_root().add_child.call_deferred(boss_instance)

func _on_boss_ready(player_instance: PlayerController, boss_instance: BossController) -> void:
	# En este punto ambos nodos están en el árbol y listos
	last_boss_health    = boss_instance.health
	last_player_health  = player_instance.health
	is_resetting = false

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
