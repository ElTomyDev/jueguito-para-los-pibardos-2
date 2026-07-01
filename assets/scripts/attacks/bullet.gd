extends CharacterBody2D
class_name Bullet

var bullet_color: Color = Color.RED

var speed: float= 800.0
var life_time: float
var dir_to_mirror: Vector2 = Vector2.ZERO
var damage: float

var from_group: StringName
var group_target: StringName
var target:Area2D = null

var hit_target: bool = false

func update(delta: float) -> void:
	_dead_if_can(delta)
	move_bullet(delta)
	queue_redraw()
	move_and_slide()

@warning_ignore("unused_parameter")
func move_bullet(delta:float) -> void:
	self.velocity = dir_to_mirror.normalized() * speed

func _dead_if_can(delta: float) -> void:
	if life_time <= 0:
		queue_free()
	life_time -= delta

func _draw() -> void:
	draw_line(Vector2.ZERO, Vector2(9,0), bullet_color, 4.0)

# Aca hay que modificar shot_impact
func delete_bullet(boss: BossController=null, player: PlayerController=null):
	hit_target = true
	if from_group == "Boss":
		if not player:
			boss.shot_impact = self.global_position
		else:
			boss.shot_impact = player.global_position
	elif from_group == "Players":
		if not boss:
			player.shot_impact = self.global_position
		else:
			player.shot_impact = boss.global_position
	queue_free()
