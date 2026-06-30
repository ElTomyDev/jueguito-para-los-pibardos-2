extends Node2D
class_name PlayerCollision

var player: PlayerController

func setup(body: PlayerController) -> void:
	player = body

func _apply_damage(damage:float) -> void:
	player.health -= damage

func _on_damage_area_body_entered(bullet: Bullet) -> void:
	if is_instance_valid(bullet):
		if bullet.is_in_group("Bullets") and bullet.group_target == "Players":
			_apply_damage(bullet.damage)
			bullet.delete_bullet(player.near_boss, player)
