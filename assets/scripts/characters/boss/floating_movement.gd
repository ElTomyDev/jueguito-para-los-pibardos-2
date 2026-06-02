extends Node2D
class_name FloatingMovement

var boss: BossController

func setup(body:BossController) -> void:
	boss = body

func update(delta: float) -> void:
	move(delta)

func move(delta: float) -> void:
	var norm_dir: Vector2 = boss.move_dir.normalized()
	if norm_dir != Vector2.ZERO:
		boss.velocity.x = lerpf(boss.velocity.x, boss.max_speed * boss.norm_dir.x, boss.acceleration_speed * delta)
		boss.velocity.y = lerpf(boss.velocity.y, boss.max_speed * boss.norm_dir.y, boss.acceleration_speed * delta)
	else:
		boss.velocity.x = lerpf(boss.velocity.x, 0.0, boss.acceleration_speed * delta)
		boss.velocity.y = lerpf(boss.velocity.y, 0.0, boss.acceleration_speed * delta)
