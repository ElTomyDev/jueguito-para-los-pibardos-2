extends Node2D
class_name Bullet

var speed: float
var life_time: float
var dispersion: float
var dir_to_mirror: Vector2
var damage: float

var group_target: StringName
var target:Area2D = null

func _ready() -> void:
	var disp_x = randf_range(-dispersion, dispersion)
	var disp_y = randf_range(-dispersion, dispersion)
	dir_to_mirror = Utils.view_to(
		self.global_position, 
		get_global_mouse_position() + Vector2(disp_x, disp_y), 
		100.0, 
		self, 
		false
	)

func _process(delta: float) -> void:
	_dead_if_can(delta)
	queue_redraw()

func _physics_process(delta: float) -> void:
	move_bullet(delta)

func move_bullet(delta:float) -> void:
	self.position += dir_to_mirror.normalized() * speed * delta 

func _dead_if_can(delta: float) -> void:
	if life_time <= 0:
		queue_free()
	life_time -= delta

func _draw() -> void:
	draw_line(Vector2.ZERO, Vector2(8,0), Color.CRIMSON, 3)

func _on_area_entered(area: Area2D) -> void:
	if area.is_in_group(group_target):
		target = area
		target.apply_damage(damage)
		queue_free()

func _on_area_exited(area: Area2D) -> void:
	if area.is_in_group(group_target):
		target = null
