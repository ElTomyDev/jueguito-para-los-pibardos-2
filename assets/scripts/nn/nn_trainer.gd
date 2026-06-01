extends Node
class_name NNTrainer

const GAMMA : float = 0.99
var nn : NeuralNetwork
var current_sigma : float = 0.5

func setup(network: NeuralNetwork) -> void:
	nn = network

func train_actor_critic(state: Array, action: Array, reward: float, next_state: Array, done: bool):
	# 1. Forward
	var outputs = nn.forward(state)
	var v_s = outputs[4]
	
	# 2. Valor Futuro
	var v_next = 0.0
	if not done:
		var next_out = nn.forward(next_state)
		v_next = next_out[4]
	
	# 3. Advantage (TD Error)
	var target = reward + (GAMMA * v_next) if not done else reward
	var advantage = target - v_s
	
	# 4. Gradiente Actor (usando advantage)
	var grad_actor = []
	var var_sq = current_sigma * current_sigma
	
	# Gradientes continuos (0,1,2)
	for i in range(3):
		grad_actor.append(-advantage * (action[i] - outputs[i]) / var_sq)
		
	# Gradiente disparo (3)
	var prob = outputs[3]
	grad_actor.append(-advantage * (action[3] - prob) / (prob * (1.0 - prob) + 1e-8))
	
	# 5. Aplicar
	nn.backprop_actor_critic(grad_actor, advantage)
