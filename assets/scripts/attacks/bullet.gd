extends CharacterBody2D
class_name Bullet

var bullet_color: Color = Color.RED

var speed: float= 300.0
var life_time: float
var dir_to_mirror: Vector2 = Vector2.ZERO
var damage: float

var boss_dir: Vector2 = Vector2.ZERO
var player_dir: Vector2 = Vector2.ZERO

var from_group: StringName
var group_target: StringName
var target:Area2D = null

func _ready() -> void:
	
	if boss_dir != Vector2.ZERO:
		# Bala del boss: usa la dirección que decidió la red
		dir_to_mirror = boss_dir 
	elif player_dir != Vector2.ZERO:
		# Bala del jugador: apunta al mouse
		dir_to_mirror = player_dir
	GlobalVars.bullets.append(self) # Agrega la bala a las variables globales para usarlas en el simulation_manager.gd

func _process(delta: float) -> void:
	_dead_if_can(delta)
	queue_redraw()

func _physics_process(delta: float) -> void:
	move_bullet(delta)
	move_and_slide()

func move_bullet(delta:float) -> void:
	self.velocity = dir_to_mirror.normalized() * speed

func _dead_if_can(delta: float) -> void:
	if life_time <= 0:
		queue_free() 
		GlobalVars.bullets.pop_at(GlobalVars.bullets.find(self))
	life_time -= delta

func _draw() -> void:
	draw_line(Vector2.ZERO, Vector2(9,0), bullet_color, 4.0)

func delete_bullet(player: PlayerController=null):
	if from_group == "Boss":
		if not player:
			GlobalVars.shot_impact = self.global_position
		else:
			GlobalVars.shot_impact = player.global_position
	GlobalVars.bullets.pop_at(GlobalVars.bullets.find(self)) # Elimina la bala de la variable global
	queue_free()


#func _on_visible_on_screen_enabler_2d_screen_exited() -> void:
#	GlobalVars.bullets.pop_at(GlobalVars.bullets.find(self))
#	queue_free()
