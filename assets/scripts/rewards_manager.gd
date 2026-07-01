extends Node2D
class_name RewardsManager

const TERM_WIN_BASE        : float = 300.0
const TERM_WIN_TIME_BONUS  : float = 100.0
const TERM_LOSE            : float = -80.0
const TERM_TIMEOUT_PEN     : float = -10.0

const R_FOR_STEP: float = -0.01

# --- Supervivencia ---
const R_DAMAGE_TAKEN_MAX   : float = -6.0
const R_DODGE_DISTANCE_MIN : float = 30.0
const R_DODGE_DISTANCE_MAX : float = 500.0
const R_ACTIVE_DODGE_GAIN   : float = 0.4   # por cada unidad de distancia que se aleja
const R_ACTIVE_DODGE_MAX    : float = 4.0  # máx por bala por frame (evita explotar)
const R_PASSIVE_DODGE       : float = 0.0   # mantener una pequeña recompensa si la bala desaparece (por si acaso)

# --- Precisión ---
const R_DAMAGE_DEALT_MAX     : float = 8.0
const R_SHOT_GOOD_AIM        : float = 6.5
const R_SHOT_HIT_PLAYER      : float = 6.5
const R_GOOD_AIM             : float = 1.0
const R_SHOT_AND_NEAR_PLAYER : float = 0.2

# --- Inactividad y movimiento ---
const R_IDLE_PENALTY        : float = -0.4
const IDLE_STREAK_THRESHOLD : int   = 10
const R_NEAR_WALLS          : float = -1.0
const R_STATIC              : float = -0.5

# --- Umbrales y margenes ---
const WALL_MARGIN      : float = 80.0
const MIN_VEL_TRESHOLD : float = 5.0

# Tracking de balas del jugador para reward de esquive
var _player_bullets_near_last_frame: Dictionary = {}

var simulation: Node2D

# Tracking de aim al momento del disparo
var last_boss_shot_step = -1
var _idle_streak = 0
var last_shot_impact: Vector2 = Vector2.ZERO

# Historial para deltas de salud
var last_player_health: float = 0.0
var last_boss_health  : float = 0.0

# Para guardad el reward promedio.
var reward_window_avg: Array = []
const MAX_REWARD_WINDOW: int = 20

func setup(body:Node2D) -> void:
	simulation = body

func calculate_reward(boss: BossController, player: PlayerController, bullets: Array) -> float:
	if not is_instance_valid(boss) or not is_instance_valid(player):
		return 0.0
	
	var p = player
	var b = boss
	var reward = 0.0
	
	# ── 0. PENALIZACION POR PASO ─────────────────────────────────────
	reward += R_FOR_STEP
	# ── 1. SUPERVIVENCIA: daño recibido ─────────────────────────────────────
	var damage_taken = last_boss_health - b.health
	if damage_taken > 0.0:
		var taken_ratio = clamp(damage_taken / b.max_health, 0.0, 1.0)
		reward += taken_ratio * R_DAMAGE_TAKEN_MAX  # negativo
	
	# ── 1b. SUPERVIVENCIA: esquivar balas del jugador ────────────────────────
	# Primero actualizamos el frame actual con dist + hit_target
	var current_player_bullets: Dictionary = {}
	for bullet in bullets:
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
	if b.shot_impact != last_shot_impact:
		var dist_impact_to_player = b.shot_impact.distance_to(p.global_position)
		if dist_impact_to_player < 80.0:
			reward += R_SHOT_HIT_PLAYER
		last_shot_impact = b.shot_impact
	
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
		var proximity_bonus = clamp(1.0 - (dist / GlobalConst.game_size.length()), 0.0, 1.0)
		reward += proximity_bonus * R_SHOT_AND_NEAR_PLAYER  # pequeño, por frame
	
	# ── Penalización por inactividad prolongada ──────────────────────────────
	if b.current_action == 0:
		_idle_streak += 1
		if _idle_streak > IDLE_STREAK_THRESHOLD:
			var idle_factor = clamp(float(_idle_streak - IDLE_STREAK_THRESHOLD) / 100.0, 0.0, 1.0)
			reward += R_IDLE_PENALTY * idle_factor
	else:
		_idle_streak = 0
	
	# ── Penalización por estar cerca de las paredes ──────────────────────────────
	var wall_dist = min(
		b.global_position.x,
		GlobalConst.game_size.x - b.global_position.x,
		b.global_position.y,
		GlobalConst.game_size.y - b.global_position.y
	)
	if wall_dist < WALL_MARGIN:
		var wall_factor = 1.0 - (wall_dist / WALL_MARGIN)
		reward += R_NEAR_WALLS * wall_factor  # proporcional, no binaria
	
	# ── Penalización por quedarse quieto ──────────────────────────────
	if b.velocity.length() < MIN_VEL_TRESHOLD:
		reward += R_STATIC
	
	# ── Actualizar tracking ──────────────────────────────────────────────────
	last_player_health = p.health
	last_boss_health   = b.health
	
	return reward

func calculate_final_reward(boss: BossController, player: PlayerController, current_step: int) -> float:
	var steps_used = float(current_step)
	var max_steps  = float(GlobalConst.MAX_STEP_FOR_EPISODE)
	var time_ratio = 1.0 - clamp(steps_used / max_steps, 0.0, 1.0)
	
	if (not is_instance_valid(player) or player.health <= 0.0):
		GlobalVars.boss_wins += 1
		return TERM_WIN_BASE + time_ratio * TERM_WIN_TIME_BONUS  # 100..130
	
	if not is_instance_valid(boss) or boss.health <= 0.0:
		var player_dmg_ratio = 0.0
		if is_instance_valid(player):
			player_dmg_ratio = 1.0 - (player.health / player.max_health)
		GlobalVars.player_wins += 1
		return TERM_LOSE + player_dmg_ratio * 20.0  # entre -100 y -60
	
	var player_hp_ratio = 1.0
	if is_instance_valid(player):
		player_hp_ratio = player.health / player.max_health
		GlobalVars.timeouts += 1
	return TERM_TIMEOUT_PEN * player_hp_ratio  # entre -30 y 0

func update_best_avg_reward(current_reward: float, current_episode: int, save_func: Callable, best_avg_reward: float=-INF, best_avg_episode: int=0) -> Dictionary:
	# Guarda la mejor recompensa
	var best_avg: float = 0.0
	var best_reward: float = best_avg_reward 
	var best_episode: int = best_avg_episode
	reward_window_avg.append(current_reward)
	if reward_window_avg.size() > MAX_REWARD_WINDOW:
		reward_window_avg.pop_front()
	
	if reward_window_avg.size() == MAX_REWARD_WINDOW:
		for r in reward_window_avg:
			best_avg += r
		best_avg = best_avg / float(reward_window_avg.size())
		
		if best_avg_reward < best_avg:
			best_reward = best_avg
			best_episode = current_episode
			save_func.call(GlobalConst.BEST_TRAIN_DATA_PATH)
	return {
		'best_reward': best_reward,
		'best_episode': best_episode
		}

func reset() -> void:
	_player_bullets_near_last_frame = {}
	last_boss_shot_step = -1
	_idle_streak = 0
	
	last_boss_health = 0.0
	last_player_health = 0.0
