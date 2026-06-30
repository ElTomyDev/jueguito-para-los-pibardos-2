extends Node

var player_wins: int = 0
var boss_wins: int = 0
var timeouts: int = 0

var player_difficulty: float = 0.0

var episode_rewards: Array = [] # Para graficar 
var best_episode_rewards: Array = []
var recent_rewards: Array = []   # guarda las últimas N recompensas
