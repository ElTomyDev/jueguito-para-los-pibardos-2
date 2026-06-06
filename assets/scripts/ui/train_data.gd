extends Control

@onready var episode_label: Label = $TrainLabels/EpisodeLabel
@onready var step_label: Label = $TrainLabels/StepLabel
@onready var reward_label: Label = $TrainLabels/RewardLabel
@onready var last_shot_impact_label: Label = $TrainLabels/LastShotImpactLabel
@onready var best_avg_reward_label: Label = $TrainLabels/BestAvgRewardLabel
@onready var best_episode_label: Label = $TrainLabels/BestEpisodeLabel
@onready var player_pos_label: Label = $TrainLabels/PlayerPosLabel
@onready var boss_pos_label: Label = $TrainLabels/BossPosLabel

@onready var move_dir_label: Label = $NNLabels/MoveDirLabel
@onready var shot_angle_label: Label = $NNLabels/ShotAngleLabel
@onready var action_label: Label = $NNLabels/ActionLabel

var view_train_info: bool = true

@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
	update_label_view()
	
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
