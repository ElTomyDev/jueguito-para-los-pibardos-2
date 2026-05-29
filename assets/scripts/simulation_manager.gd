extends Node2D

@export var players:Array[PackedScene]
@export var spawn_player_points:Array[Node2D]
@export var boss_spawn:Node2D
@export var boss: PackedScene

func _ready() -> void:
	spawn_entities()

func spawn_entities() -> void:
	for idx in range(len(players)):
		var player_instance = players[idx].instantiate()
		player_instance.global_position = spawn_player_points[idx].global_position
		add_child(player_instance)
	var boss_instance = boss.instantiate()
	boss_instance.global_position = boss_spawn.global_position
	add_child(boss_instance)
