extends CanvasLayer

@onready var reward_graph: Graph2D = $Control/Graph2D

@onready var episode_label: Label = $Control/TrainLabels/EpisodeLabel
@onready var step_label: Label = $Control/TrainLabels/StepLabel
@onready var reward_label: Label = $Control/TrainLabels/RewardLabel
@onready var last_shot_impact_label: Label = $Control/TrainLabels/LastShotImpactLabel
@onready var best_avg_reward_label: Label = $Control/TrainLabels/BestAvgRewardLabel
@onready var best_episode_label: Label = $Control/TrainLabels/BestEpisodeLabel
@onready var player_pos_label: Label = $Control/TrainLabels/PlayerPosLabel
@onready var boss_pos_label: Label = $Control/TrainLabels/BossPosLabel
@onready var boss_win_label: Label = $Control/TrainLabels/BossWinLabel
@onready var player_win_label: Label = $Control/TrainLabels/PlayerWinLabel
@onready var timeouts_label: Label = $Control/TrainLabels/TimeoutsLabel
@onready var player_difficulty_label: Label = $Control/TrainLabels/PlayerDifficultyLabel

@onready var move_dir_label: Label = $Control/NNLabels/MoveDirLabel
@onready var shot_angle_label: Label = $Control/NNLabels/ShotAngleLabel
@onready var action_label: Label = $Control/NNLabels/ActionLabel

var episode_rewards: Array = []
var reward_plot: PlotItem
var best_reward_plot: PlotItem

var view_graph: bool = true
var view_train_info: bool = true
var graph_x_margin: int = 10

var update_rate: int = 10

var _last_plotted_episode: int = -1
var _ui_frame_counter: int = 0

@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	update_info_view()
	_ui_frame_counter += 1
	if view_graph:
		reward_graph.visible = true
		update_reward_graph()
	else:
		reward_graph.visible = false
	
	if view_train_info and _ui_frame_counter % 6 == 0:
		update_train_labels()
		update_nn_labels()

func update_train_labels() -> void:
	episode_label.text = "Episode: %d" % [GlobalVars.current_episode]
	step_label.text = "Step: %d" % [GlobalVars.current_step]
	reward_label.text = "Reward: %d" % [GlobalVars.current_reward]
	last_shot_impact_label.text = "Shot Impact To Player: (%d, %d)" % [GlobalVars.shot_impact.x, GlobalVars.shot_impact.y]
	best_avg_reward_label.text = "Best Avg Reward: %.2f" % [GlobalVars.best_avg_reward]
	best_episode_label.text = "Best Episode: %d" % [GlobalVars.best_avg_episode]
	boss_win_label.text = "Boss Wins: %d" % [GlobalVars.boss_wins]
	player_win_label.text = "Player Wins: %d" % [GlobalVars.player_wins]
	timeouts_label.text = "Timeouts: %d" % [GlobalVars.timeouts]
	player_difficulty_label.text = "Player difficulty: %.4f" % [GlobalVars.player_difficulty]
	if is_instance_valid(GlobalVars.players[0]): 
		player_pos_label.text = "Player Position: (%d, %d)" % [GlobalVars.players[0].global_position.x, GlobalVars.players[0].global_position.y]
	if is_instance_valid(GlobalVars.boss):
		boss_pos_label.text = "Boss Position: (%d, %d)" % [GlobalVars.boss.global_position.x, GlobalVars.boss.global_position.y]
	
func update_nn_labels() -> void:
	move_dir_label.text = "Move Dir[0][1]: (%.2f, %.2f)" % [GlobalVars.nn_outputs["move_dir"][0], GlobalVars.nn_outputs["move_dir"][1]]
	shot_angle_label.text = "Shot Angle[2]: %.2f" % [GlobalVars.nn_outputs["shot_angle"]]
	action_label.text = "Action[3]: %d" % [GlobalVars.nn_outputs["action"]]

func update_info_view() -> void:
	if Input.is_action_just_pressed("toggle_train_info"):
		for label in $Control/TrainLabels.get_children():
			label.visible = !label.visible
		view_train_info = !view_train_info
	if Input.is_action_just_pressed("toggle_graphs"):
		view_graph = !view_graph

func update_reward_graph() -> void:
	if reward_graph and not reward_plot:
		reward_plot = reward_graph.add_plot_item("", Color.BLUE)
		best_reward_plot = reward_graph.add_plot_item("", Color.RED)
		# Carga inicial de todos los puntos existentes
		for i in GlobalVars.episode_rewards.size():
			if i % graph_x_margin == 0:
				reward_plot.add_point(Vector2(i, GlobalVars.episode_rewards[i]))
		for i in GlobalVars.best_episode_rewards.size():
			if i % graph_x_margin == 0:
				best_reward_plot.add_point(Vector2(i, GlobalVars.best_episode_rewards[i]))
		_last_plotted_episode = GlobalVars.episode_rewards.size() - 1
		return
	
	# Solo agrega los puntos nuevos
	var current_size = GlobalVars.episode_rewards.size()
	if current_size > 0 and _last_plotted_episode < current_size - 1:
		for i in range(_last_plotted_episode + 1, current_size):
			if i % graph_x_margin == 0:
				reward_plot.add_point(Vector2(i, GlobalVars.episode_rewards[i]))
		for i in range(_last_plotted_episode + 1, current_size):
			if i % graph_x_margin == 0:
				best_reward_plot.add_point(Vector2(i, GlobalVars.best_episode_rewards[i]))
		_last_plotted_episode = current_size - 1
