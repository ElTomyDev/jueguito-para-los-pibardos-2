extends Node

var current_reward: float = 0.0
var current_episode: int = 0
var current_step: int = 0
var MAX_STEP_FOR_EPISODE: int = 0

var players: Array[PlayerController] = []
var bullets: Array[Bullet] = []
var boss: BossController = null

# Ultimo impacto de bala echo por el jefe
var shot_impact: Vector2 = Vector2.ZERO

var nn_outputs: Dictionary = {}
