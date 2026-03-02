"""
Pseudocode for the environment

init():

    1. generate reference questionnaire output and corpus logprobs

    2. put them using ray to make available for other nodes

    3. spin up vLLMWorkers (one LLM per GPU, each has to load reference data in __init__ and initialize LLM)

step(action: batch of weight perturbations):

    1. broadcast weight update to LLM using ray

    2. vLLM worker generates questionnaire responses, scores them and computes KL divergence, returns single reward

    3. return rewards (one per perturbation set)

"""

"""
from ray.util.actor_pool import ActorPool

# Initialize your pool with the 23 workers
pool = ActorPool(self.workers)

def step(self, large_batch_of_perturbations):
    # large_batch_of_perturbations: list of 115 tensors
    
    # 1. Map the perturbations to the pool
    # Ray will feed these to the 23 GPUs as they become available
    # It handles the "wait until a GPU is free" logic for you
    rewards_generator = pool.map(
        lambda worker, delta: worker.process_episode.remote(delta),
        large_batch_of_perturbations
    )
    
    # 2. Collect all 115 rewards as they finish
    all_rewards = list(rewards_generator)
    
    return all_rewards
"""