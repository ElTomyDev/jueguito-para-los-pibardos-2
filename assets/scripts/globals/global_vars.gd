extends Node

# Informacion sobre entrenamiento
var current_reward: float = 0.0
var current_episode: int = 0
var current_step: int = 0

var recent_rewards: Array = []   # guarda las últimas N recompensas
var best_avg_reward: float = -1e9
var best_avg_episode: int = 0

var players: Array[PlayerController] = []
var bullets: Array[Bullet] = []
var boss: BossController = null

# Ultimo impacto de bala echo por el jefe
var shot_impact: Vector2 = Vector2.ZERO

var nn_outputs: Dictionary = {}
