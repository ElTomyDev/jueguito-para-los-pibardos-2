extends Control

@onready var episode_label: Label = $Labels/EpisodeLabel
@onready var step_label: Label = $Labels/StepLabel
@onready var reward_label: Label = $Labels/RewardLabel

var last_shot_point_position : Vector2 = Vector2.ZERO
var shot_point_labels: Array[Label] = []

func _process(delta: float) -> void:
	episode_label.text = "Episode: %d" % [GlobalVars.current_episode]
	step_label.text = "Step: %d" % [GlobalVars.current_step]
	reward_label.text = "Reward: %d" % [GlobalVars.current_reward]
	_draw_shot_point()
	queue_redraw()

func _draw_shot_point() -> void:
	var label_instance: Label = Label.new()
	if GlobalVars.shot_impact != last_shot_point_position and GlobalVars.shot_impact != Vector2.ZERO:
		draw_circle(GlobalVars.shot_impact, 5, Color(Color.RED, 0.3), false, 20)
		label_instance.text = "(%.2f, %.2f)" % [GlobalVars.shot_impact.x, GlobalVars.shot_impact.y]
		label_instance.global_position = GlobalVars.shot_impact
		label_instance.add_theme_font_size_override("font_size", 8)
		last_shot_point_position = GlobalVars.shot_impact
	shot_point_labels.append(label_instance)
	get_tree().get_root().add_child(label_instance)
	_delete_all_shot_impacts()

func _draw() -> void:
	if GlobalVars.shot_impact != last_shot_point_position and GlobalVars.shot_impact != Vector2.ZERO:
		draw_circle(GlobalVars.shot_impact, 5, Color(Color.RED, 0.3))

func _delete_all_shot_impacts() -> void:
	if shot_point_labels.is_empty(): return
	if GlobalVars.current_step >= GlobalVars.MAX_STEP_FOR_EPISODE:
		for label in shot_point_labels:
			label.queue_free()
		shot_point_labels.clear()
