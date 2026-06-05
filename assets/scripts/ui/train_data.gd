extends Control

@onready var episode_label: Label = $Labels/EpisodeLabel
@onready var step_label: Label = $Labels/StepLabel
@onready var reward_label: Label = $Labels/RewardLabel
@onready var last_shot_impact_label: Label = $Labels/LastShotImpactLabel
@onready var best_avg_reward_label: Label = $Labels/BestAvgRewardLabel
@onready var best_episode_label: Label = $Labels/BestEpisodeLabel
@onready var player_pos_label: Label = $Labels/PlayerPosLabel
@onready var boss_pos_label: Label = $Labels/BossPosLabel

@warning_ignore("unused_parameter")
func _process(delta: float) -> void:
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
	
