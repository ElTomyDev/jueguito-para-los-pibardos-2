extends Node
class_name NNTrainer

const GAMMA : float = 0.99
var nn : NeuralNetwork
var current_sigma : float = 0.5

func setup(network: NeuralNetwork) -> void:
	nn = network

func train_actor_critic(state: Array, action: Array, reward: float, next_state: Array, done: bool):
	var outputs = nn.forward(state)
	var v_s = outputs[4]
	
	var v_next = 0.0
	if not done:
		var next_out = nn.forward(next_state)
		v_next = next_out[4]
	
	var advantage = reward + (GAMMA * v_next) - v_s
	
	var grad_actor = []
	var var_sq = current_sigma * current_sigma
	
	# Gradientes Actor
	for i in range(3):
		grad_actor.append((action[i] - outputs[i]) / var_sq)
		
	var prob = clamp(outputs[3], 0.01, 0.99)
	grad_actor.append((action[3] - prob) / (prob * (1.0 - prob)))
	
	nn.backprop_actor_critic(grad_actor, advantage)
