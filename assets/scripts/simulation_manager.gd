extends Node2D
class_name SimulationManager

@export var player_scene       : PackedScene
@export var boss_scene         : PackedScene
@export var player_spawn_point : Node2D
@export var boss_spawn_point   : Node2D
@export var random_spawns      : bool = false
@export var load_best_model    : bool = true
@export var is_new_train       : bool = false

var current_reward: float = 0.0
var best_avg_reward: float = -INF
var best_avg_episode: int = 0

var rewards_manager: RewardsManager

var boss: BossController = null
var player: PlayerController = null
var bullets: Array[Bullet] = []

var current_episode: int = 0
var current_step: int = 0

# NN
var nn_client: NNClient
var nn_outputs: Dictionary = {
	'move_dir':[0.0, 0.0],
	'shot_angle': 0.0,
	'action': 0
	}

# Para aumentar dificultad de jugador automatico.
var _boss_win_rate_window: Array = []  # últimos N episodios
const DIFFICULTY_WINDOW: int = 50

var is_resetting: bool = false

func _ready() -> void:
	_init_simulation()

@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	toggle_auto_player()

func _physics_process(delta: float) -> void:
	# Verificaciones antes de correr un paso
	if is_resetting: return
	if not is_instance_valid(boss): return
	if player.is_empty(): return

	nn_client.poll()
	
	var current_inputs: Array = _get_inputs_for_nn(boss)
	var reward: float  = rewards_manager.calculate_reward(boss, player, bullets)
	current_reward += reward
	
	# Solo enviar si no hay una petición activa
	if not nn_client.is_busy():
		nn_client.request_action(current_inputs, reward)
	
	var response = nn_client.get_last_action()
	nn_outputs["move_dir"]   = response.get("move_dir",   [0.0, 0.0])
	nn_outputs["shot_angle"] = response.get("shot_angle", 0.0)
	nn_outputs["action"]     = response.get("action",     0)
	_update_step(delta)
	current_step += 1
	if _can_episode_end():
		_handle_episode_end()

func _update_step(delta) -> void:
	boss.update(delta, nn_outputs, [player])
	player.update(delta, boss, bullets, current_episode)
	for b in bullets:
		b.update(delta)

func _init_simulation() -> void:
	Engine.time_scale = 1.0
	Engine.max_physics_steps_per_frame = 1
	nn_client = NNClient.new()
	_load_train_data(GlobalConst.BEST_TRAIN_DATA_PATH if load_best_model else GlobalConst.TRAIN_DATA_PATH)
	_reset_episode()

# Arreglar player_dificuly
func _handle_episode_end() -> void:
	is_resetting = true
	
	var boss_win: bool = (not is_instance_valid(player) and player.health <= 0.0)
	
	# Actualizar ventana de win rate ANTES de calcular final reward
	_boss_win_rate_window.append(boss_win)
	if _boss_win_rate_window.size() > DIFFICULTY_WINDOW:
		_boss_win_rate_window.pop_front()
	if _boss_win_rate_window.size() >= DIFFICULTY_WINDOW:
		update_difficulty(_boss_win_rate_window)
	
	var timed_out: bool = current_step >= GlobalConst.MAX_STEP_FOR_EPISODE
	var final_reward: float = rewards_manager.calculate_final_reward(boss, player, current_step)
	nn_client.notify_episode_end(
		current_episode,
		current_reward,
		final_reward,
		timed_out,
		boss_win
	)
	
	_save_train_data(GlobalConst.TRAIN_DATA_PATH)
	var best_info_dict = rewards_manager.update_best_avg_reward(current_reward, current_episode, _save_train_data ,best_avg_reward, best_avg_episode)
	best_avg_reward = best_info_dict['best_reward']
	best_avg_episode = best_info_dict['best_episode']
	current_episode += 1
	GlobalVars.best_episode_rewards.append(best_avg_reward)
	GlobalVars.episode_rewards.append(current_reward)
	
	_reset_episode()

func update_difficulty(boss_win_rate_window: Array) -> void:
	# Calcula win rate del boss_scene en la ventana
	var wins = boss_win_rate_window.count(true)
	var win_rate = float(wins) / float(boss_win_rate_window.size())
	
	# Solo sube la dificultad si el boss_scene gana más del N%
	if win_rate > 0.19:
		GlobalVars.player_difficulty = clamp(GlobalVars.player_difficulty + 0.02, 0.0, 1.0)
	elif win_rate < 0.09:
		GlobalVars.player_difficulty = clamp(GlobalVars.player_difficulty - 0.01, 0.0, 1.0)



# ------
# Inputs
# ------
func _get_inputs_for_nn(boss_i: BossController) -> Array:
	if not is_instance_valid(boss_i):
		push_error("Para obtener los inputs debe existir un 'BossController'.")
	var dist_to_player  = 1.0
	var dist_to_mouse: float = 1.0
	var angle_to_player_raw: float = 0.0  # ángulo directo al jugador, normalizado
	var player_vel       = Vector2.ZERO
	var rel_vel          = Vector2.ZERO
	var time_since_last: float
	
	
	if is_instance_valid(boss_i.near_player):
		dist_to_player = clamp(
			boss_i.global_position.distance_to(boss_i.near_player.global_position) / GlobalConst.game_size.length(),
			0.0, 1.0
		)
		var to_player = (boss_i.near_player.global_position - boss_i.global_position).angle()
		angle_to_player_raw = to_player / PI  # normalizado [-1, 1]
		player_vel  = boss_i.near_player.velocity.normalized()
		rel_vel     = boss_i.near_player.velocity - boss_i.velocity
	
	var src_last_shot = boss_i.shot_attack.last_shot_step if is_instance_valid(boss_i.shot_attack) else -1
	if src_last_shot <= 0:
		time_since_last = 1.0
	else:
		time_since_last = clamp(
			float(GlobalVars.current_step - src_last_shot) / float(GlobalConst.MAX_STEP_FOR_EPISODE),
			0.0, 1.0
		)
	
	var dist_to_center = boss_i.global_position.distance_to(GlobalConst.game_size / 2) / GlobalConst.game_size.length()
	var b_dist: Array = []
	var b_pos:  Array = []
	var b_vel:  Array = []
	var b_dir_to_boss_y:  Array = []
	var b_dir_to_boss_x:  Array = []
	var b_approach: Array = []
	for bullet in boss_i.bullets_detected:
		if is_instance_valid(bullet):
			var b_to_boss_x = (boss_i.global_position.x - bullet.global_position.x) / GlobalConst.game_size.x
			var b_to_boss_y = (boss_i.global_position.y - bullet.global_position.y) / GlobalConst.game_size.y
			var to_boss_dir = Vector2(b_to_boss_x, b_to_boss_y).normalized()
			var approach_vel = bullet.velocity.dot((boss_i.global_position - bullet.global_position).normalized()) / bullet.speed
			b_approach.append(clamp(approach_vel, -1.0, 1.0))
			b_dir_to_boss_x.append(to_boss_dir.x)  # ya en [-1, 1]
			b_dir_to_boss_y.append(to_boss_dir.y)
			b_dist.append(clamp(boss_i.global_position.distance_to(bullet.global_position) / GlobalConst.game_size.length(), 0.0, 1.0))
			b_pos.append(bullet.global_position.x / GlobalConst.game_size.x)
			b_pos.append(bullet.global_position.y / GlobalConst.game_size.y)
			b_vel.append(bullet.velocity.x / bullet.speed)
			b_vel.append(bullet.velocity.y / bullet.speed)
		else:
			b_dist.append(1.0)
			b_pos.append(0.0)
			b_pos.append(0.0)
			b_vel.append(0.0)
			b_vel.append(0.0)
			b_dir_to_boss_x.append(0.0)
			b_dir_to_boss_y.append(0.0)
			b_approach.append(0.0)
	
	var mouse_pos: Vector2 = get_global_mouse_position()
	dist_to_mouse = clamp(
			global_position.distance_to(mouse_pos) / GlobalConst.game_size.length(),
			0.0, 1.0
		)
	
	var inputs = [
		boss_i.global_position.x / GlobalConst.game_size.x,
		boss_i.global_position.y / GlobalConst.game_size.y,
		boss_i.shot_impact.x / GlobalConst.game_size.x,
		boss_i.shot_impact.y / GlobalConst.game_size.y,
		time_since_last,
		boss_i.velocity.x / boss_i.max_speed,
		boss_i.velocity.y / boss_i.max_speed,
		boss_i.health / boss_i.max_health,
		dist_to_player,
		angle_to_player_raw,  
		clamp(rel_vel.x / boss_i.max_speed, -1.0, 1.0),
		clamp(rel_vel.y / boss_i.max_speed, -1.0, 1.0),
		player_vel.x,
		player_vel.y,
		dist_to_center,
		float(boss_i.current_action)/ len(boss_i.current_action),                                                                                                                        
		clamp(current_step / float(GlobalConst.MAX_STEP_FOR_EPISODE), 0.0, 1.0),
		clamp(boss_i.damage / boss_i.max_damage, 0.0, 1.0),
		mouse_pos.x / GlobalConst.game_size.x,
		mouse_pos.y / GlobalConst.game_size.y,
		dist_to_mouse
	]
	inputs.append_array(b_dist)
	inputs.append_array(b_pos)
	inputs.append_array(b_vel)
	inputs.append_array(b_dir_to_boss_x)
	inputs.append_array(b_dir_to_boss_y)
	inputs.append_array(b_approach)
	return inputs

# ------------
# Utilidad
# ------------
func _reset_episode() -> void:
	for bullet in bullets:
		if is_instance_valid(bullet): bullet.queue_free()
	if is_instance_valid(player): player.queue_free()
	if is_instance_valid(boss): boss.queue_free()
	
	boss = null
	player = null
	bullets.clear()
	current_step  = 0
	current_reward = 0.0
	nn_outputs["move_dir"]   = [0.0, 0.0]
	nn_outputs["shot_angle"] = 0.0
	nn_outputs["action"]     = 0
	_spawn_entities()

func _spawn_entities() -> void:
	var player_instance = player_scene.instantiate() as PlayerController
	var boss_instance   = boss_scene.instantiate() as BossController
	
	var entities_pos: Array = _get_entities_position() 
	player_instance.global_position = entities_pos[0]
	boss_instance.global_position   = entities_pos[1]
	
	player = player_instance.init()
	boss = boss_instance.init()

	get_tree().get_root().add_child.call_deferred(player_instance)
	get_tree().get_root().add_child.call_deferred(boss_instance)

func _get_entities_position() -> Array:
	var player_pos: Vector2
	var boss_pos: Vector2
	if not random_spawns:
		player_pos = player_spawn_point.global_position
		boss_pos = boss_spawn_point.global_position
	else:
		var margin_spawn: float = 79.0
		var margin: float  = 199.0
		var max_attempts   = 19
		for _i in range(max_attempts):
			player_pos = Vector2(
				randf_range(margin_spawn, GlobalConst.game_size.x - margin_spawn),
				randf_range(margin_spawn, GlobalConst.game_size.y - margin_spawn)
			)
			boss_pos   = Vector2(
				randf_range(margin_spawn, GlobalConst.game_size.x - margin_spawn),
				randf_range(margin_spawn, GlobalConst.game_size.y - margin_spawn)
			)
			if player_pos.distance_to(boss_pos) >= margin:
				break
	return [player_pos, boss_pos]

func _can_episode_end() -> bool:
	if not is_instance_valid(boss): return true
	if not is_instance_valid(player): return true
	return (
		current_step >= GlobalConst.MAX_STEP_FOR_EPISODE
		or boss.health   <= 0.0
		or player.health <= 0.0
	)

func _load_train_data(path) -> void:
	if not is_new_train:
		var data = ExternalFileManager.read_json(path)
		if data.is_empty():
			print("No hay datos para cargar: ", path)
			return
		current_episode   = data.get('episode', 0)
		best_avg_reward   = data.get('best_avg_reward', -INF)
		best_avg_episode  = data.get('best_avg_episode', 0)
		GlobalVars.player_wins = data.get('player_wins', 0)
		GlobalVars.boss_wins = data.get('boss_wins', 0)
		GlobalVars.timeouts = data.get('timeouts', 0)
		GlobalVars.player_difficulty = data.get('player_difficulty', 0.0)
		GlobalVars.episode_rewards   = data.get('episodes_rewards', [])
		GlobalVars.best_episode_rewards = data.get('best_episode_rewards', [])

func _save_train_data(path) -> void:
	var data: Dictionary = {
		'episode':        current_episode,
		'best_avg_reward':  best_avg_reward,
		'best_avg_episode': best_avg_episode,
		'player_wins': GlobalVars.player_wins,
		'boss_wins': GlobalVars.boss_wins,
		'timeouts': GlobalVars.timeouts,
		'player_difficulty':  GlobalVars.player_difficulty,
		'episodes_rewards': GlobalVars.episode_rewards,
		'best_episode_rewards': GlobalVars.best_episode_rewards,
	}
	ExternalFileManager.save_data(data, path)

func toggle_auto_player() -> void:
	if not is_instance_valid(player): return
	
	if Input.is_action_just_pressed("is_automatic_player"):
		player.is_automatic = !player.is_automatic
