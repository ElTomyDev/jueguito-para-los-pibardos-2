extends Control

@onready var episode_label: Label = $Labels/EpisodeLabel
@onready var step_label: Label = $Labels/StepLabel
@onready var reward_label: Label = $Labels/RewardLabel



func _process(delta: float) -> void:
	episode_label.text = "Episode: %d" % [GlobalVars.current_episode]
	step_label.text = "Step: %d" % [GlobalVars.current_step]
	reward_label.text = "Reward: %d" % [GlobalVars.current_reward]
