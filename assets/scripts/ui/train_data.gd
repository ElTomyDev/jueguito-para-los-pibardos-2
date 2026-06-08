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

@onready var move_dir_label: Label = $Control/NNLabels/MoveDirLabel
@onready var shot_angle_label: Label = $Control/NNLabels/ShotAngleLabel
@onready var action_label: Label = $Control/NNLabels/ActionLabel

var episode_rewards: Array = []
var reward_plot: PlotItem

var view_graph: bool = true
var view_train_info: bool = true

@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	update_label_view()
	if view_graph:
		reward_graph.visible = true
		update_reward_graph()
	else:
		reward_graph.visible = false
	if view_train_info:
		update_train_labels()
		update_nn_labels()

func update_train_labels() -> void:
	episode_label.text = "Episode: %d" % [GlobalVars.current_episode]
	step_label.text = "Step: %d" % [GlobalVars.current_step]
	reward_label.text = "Reward: %d" % [GlobalVars.current_reward]
	last_shot_impact_label.text = "Shot Impact To Player: (%d, %d)" % [GlobalVars.shot_impact.x, GlobalVars.shot_impact.y]
	best_avg_reward_label.text = "Best Avg Reward: %.2f" % [GlobalVars.best_avg_reward]
	best_episode_label.text = "Best Episode: %d" % [GlobalVars.best_avg_episode]
	if is_instance_valid(GlobalVars.players[0]): 
		player_pos_label.text = "Player Position: (%d, %d)" % [GlobalVars.players[0].global_position.x, GlobalVars.players[0].global_position.y]
	if is_instance_valid(GlobalVars.boss):
		boss_pos_label.text = "Boss Position: (%d, %d)" % [GlobalVars.boss.global_position.x, GlobalVars.boss.global_position.y]

func update_nn_labels() -> void:
	move_dir_label.text = "Move Dir[0][1]: (%.2f, %.2f)" % [GlobalVars.nn_outputs["move_dir"][0], GlobalVars.nn_outputs["move_dir"][1]]
	shot_angle_label.text = "Shot Angle[2]: %.2f" % [GlobalVars.nn_outputs["shot_angle"]]
	action_label.text = "Action[3]: %d" % [GlobalVars.nn_outputs["action"]]

func update_label_view() -> void:
	if Input.is_action_just_pressed("toggle_train_info"):
		view_train_info = !view_train_info
	if Input.is_action_just_pressed("toggle_graphs"):
		view_graph = !view_graph
	
func update_reward_graph() -> void:
	if reward_graph:
		if Input.is_action_just_pressed("g_right"):
			reward_graph.x_max += 10
		if Input.is_action_just_pressed("g_left"):
			reward_graph.x_max -= 10
		if Input.is_action_just_pressed("g_up"):
			reward_graph.y_min -= 10
			reward_graph.y_max += 10
		if Input.is_action_just_pressed("g_down"):
			reward_graph.y_min += 10
			reward_graph.y_max -= 10
	# 1. Initialize the plot the first time the function is called
	if reward_graph and not reward_plot:
		# 'add_plot_item' returns a PlotItem object, which we store to reference the plot later
		reward_plot = reward_graph.add_plot_item("", Color.YELLOW)
	
	if not GlobalVars.episode_rewards.is_empty():
		# 3. Update the graph: clear the old points and add the new series
		if reward_plot:
			# Clear all points from the existing plot
			reward_plot.remove_all()
			
			# Add all accumulated rewards as points again
			for i in GlobalVars.episode_rewards.size():
				reward_plot.add_point(Vector2(i, GlobalVars.episode_rewards[i]))
