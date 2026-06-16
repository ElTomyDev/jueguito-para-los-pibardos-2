extends Node

# Informacion sobre entrenamiento
var current_reward: float = 0.0
var current_episode: int = 0
var current_step: int = 0

var player_wins: int = 0
var boss_wins: int = 0
var timeouts: int = 0

var episode_rewards: Array = [] # Para graficar 
var best_episode_rewards: Array = []

var recent_rewards: Array = []   # guarda las últimas N recompensas
var best_avg_reward: float = -1e9
var best_avg_episode: int = 0

var players: Array[PlayerController] = []
var bullets: Array[Bullet] = []
var boss: BossController = null

# Ultimo impacto de bala echo por el jefe
var shot_impact: Vector2 = Vector2.ZERO

var nn_outputs: Dictionary = {
			"move_dir"      : [0.0,0.0],
			"shot_angle"    : 0.0,
			"action"        : 0,
		}
