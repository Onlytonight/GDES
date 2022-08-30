import os
from time import perf_counter
import numpy as np
import gym
import ES.actorES32 as actor
import paddle
import ES.optimizers as optimizers
import json
import pickle
from mpi4py import MPI
from util.utils import save_reward

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
ENV_NAME = 'GraphEnv-v1'
graph_topology = 3
NUM_ACTIONS = 4  # We limit the actions to the K=4 shortest paths

# MPI
mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()  # Number of processes in the comm
mpi_rank = mpi_comm.Get_rank()  # ID of process within the comm

# RNG
SEED = 9
os.environ['PYTHONHASHSEED'] = str(SEED)


def mpi_print(*args, **kwargs):
    if mpi_rank == 0:
        print(*args, **kwargs)


def old_cummax(alist, extractor):
    with paddle.static.name_scope('cummax'):
        maxes = [paddle.fluid.layers.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [paddle.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(paddle.add_n(maxes[0:i + 1]))
    return cummaxes


class PerturbationsGenerator:
    def __init__(self, number_params):
        self.number_params = number_params
        # Initialize RNG
        if mpi_rank:
            self.RNG = np.random.RandomState(mpi_rank)
        else:
            self.RNG = np.array([np.random.RandomState(ii) for ii in range(mpi_size)], dtype=object)

    def obtain_perturbations(self, perturbations):
        # Generate  perturbations
        return self.RNG.randn(perturbations, self.number_params)

    def global_obtain_perturbations(self, episodes_per_worker, episodes_indices, total_episodes):
        global_res = np.empty((total_episodes, self.number_params), dtype=np.float32)
        local_res = None
        for ii, episodes in enumerate(list(episodes_per_worker)):
            if episodes:
                temp_res = self.RNG[ii].randn(episodes, self.number_params)
                if not ii:
                    local_res = temp_res
                global_res[episodes_indices[ii]: episodes_indices[ii + 1]] = temp_res
        return local_res, global_res


class ESPolicy:
    def __init__(self,
                 hparams,
                 listofDemands,
                 params_file,
                 param_noise,
                 action_noise,
                 number_mutations,
                 model
                 ):

        hidden_init_actor = paddle.nn.initializer.Orthogonal(gain=np.sqrt(2))
        kernel_init_actor = paddle.nn.initializer.Orthogonal(gain=np.sqrt(0.01))

        if model is None:
            self.actor = actor.myModel(hparams, hidden_init_actor, kernel_init_actor)
        else:
            self.actor = model
        self.actor_hparams = hparams
        # Current set of weights
        self.current_weights = self.actor.get_weights()
        # Random number stream
        self.policy_rng = np.random.RandomState(1)
        # Enviroments
        self.tr_envs = list()
        self.tr_envs.append(gym.make(ENV_NAME))
        self.tr_envs[-1].seed(SEED)
        self.tr_envs[-1].generate_environment(graph_topology, listofDemands)

        self.current_tr_environment = None
        self.eval_envs = list()
        self.eval_envs.append(gym.make(ENV_NAME))
        self.eval_envs[-1].seed(SEED)
        self.eval_envs[-1].generate_environment(graph_topology, listofDemands)

        # Parameters for policy rollout
        self.ACTION_NOISE_STD = action_noise
        self.PARAM_NOISE_STD = param_noise

        if params_file:
            try:
                with open("saved_params/{}.pkcl".format(params_file), 'rb') as ff:
                    self.current_weights = pickle.load(ff)
                self.actor.set_weights(self.current_weights)
            except FileNotFoundError as err:
                # The parameters haven't been saved until now: we store them for future reference
                with open("saved_params/{}.pkcl".format(params_file), 'wb') as ff:
                    pickle.dump(self.current_weights, ff)
        else:
            mpi_print("Warning: No hyperparams file detected!")

        # Total Number of mutations
        self.total_number_mutations = number_mutations

    def tr_rollout(self):

        # Accumulator of rewards
        total_rewards = 0.0
        # Enviroment variables (State)
        graph_state, demand, source, destination = self.tr_envs[self.current_tr_environment].reset()
        done = False
        # Iterate through the enviroment
        while not done:
            action_dist = self.pred_action_node_distrib(self.tr_envs[self.current_tr_environment], source, destination,
                                                        demand)
            # If needed add noise to the action distribution
            if self.ACTION_NOISE_STD > 0:
                action_dist += self.policy_rng.randn(action_dist.size) * self.ACTION_NOISE_STD
                action_dist[action_dist < 0] = 0  # We remove negative terms possibly caused by noise
                action_dist /= action_dist.sum()
            action = action_dist.argmax()
            # Allocate the traffic of the demand to the shortest path
            graph_state, reward, done, demand, source, destination = self.tr_envs[
                self.current_tr_environment].make_step(graph_state,
                                                       action, demand, source, destination)

            total_rewards += reward
        # Return aggregated results
        return total_rewards

    def perform_tr_rollouts(self, perturbations, iteration):
        # If we have no perturbations, perform no rollouts
        if perturbations.shape[0] == 0:
            return NULL_BUFF, NULL_BUFF

        # Select current environment
        self.current_tr_environment = iteration % len(self.tr_envs)

        # Define rewards buffer
        positive_rewards = np.empty(perturbations.shape[0], dtype=np.float32)
        negative_rewards = np.empty_like(positive_rewards)

        for ii in range(positive_rewards.size):
            noise = perturbations[ii] * self.PARAM_NOISE_STD
            # Positive noise
            self.actor.set_weights(self.current_weights + noise)
            positive_rewards[ii] = self.tr_rollout()
            # Negative reward
            self.actor.set_weights(self.current_weights - noise)
            negative_rewards[ii] = self.tr_rollout()

        # Reset weights
        self.actor.set_weights(self.current_weights)
        # Obtain both rewards per rank
        return positive_rewards, negative_rewards

    def eval_rollout(self, env):
        # List of rewards
        total_rewards = 0.0
        # Enviroment variables (State)
        graph_state, demand, source, destination = env.reset()
        done = False
        # Iterate through the enviroment
        while not done:
            action_dist = self.pred_action_node_distrib(env, source, destination, demand)
            # If needed add noise to the action distribution
            action = action_dist.argmax()
            # Allocate the traffic of the demand to the shortest path
            graph_state, reward, done, demand, source, destination = env.make_step(graph_state, action, demand, source,
                                                                                   destination)
            total_rewards += reward
        # Return aggregated results
        return total_rewards

    def perform_eval_rollouts(self, evaluation_episodes, file_name):
        if evaluation_episodes == 0:
            return NULL_BUFF
        local_eval_rewards = np.empty((evaluation_episodes, len(self.eval_envs)), dtype=np.float32)
        for eval_iter in range(evaluation_episodes):
            local_eval_rewards[eval_iter] = self.eval_rollout(self.eval_envs[-1])
        # Return the local rewards
        return local_eval_rewards

    def pred_action_node_distrib(self, env, source, destination, demand):
        """
        Method to obtain the action distribution
        """
        # List of graph features that are used in the cummax() call
        list_k_features = list()

        k_path = 0

        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        while k_path < len(env.allPaths[str(source) + ':' + str(destination)]):
            env.mark_action_k_path(k_path, source, destination, demand)

            features = self.get_graph_features(env, source, destination)
            list_k_features.append(features)

            # We desmark the bw_allocated
            env.graph_state[:, 1] = 0
            k_path = k_path + 1

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.
        graph_ids = [paddle.vision.transforms.Pad([paddle.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = old_cummax(vs, lambda v: v['first'])
        second_offset = old_cummax(vs, lambda v: v['second'])

        tensor = ({
            'graph_id': paddle.fluid.layers.concat([v for v in graph_ids], axis=0),
            'link_state': paddle.fluid.layers.concat([v['link_state'] for v in vs], axis=0),
            'first': paddle.fluid.layers.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': paddle.fluid.layers.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': paddle.add_n([v['num_edges'] for v in vs]),
        }
        )
        # Predict action probabilities (i.e., one per graph/action)
        r = self.actor(tensor['link_state'], tensor['graph_id'], tensor['first'], tensor['second'],
                       tensor['num_edges'], training=False)
        listQValues = paddle.fluid.layers.reshape(r, (1, len(r)))
        softMaxQValues = paddle.nn.functional.softmax(listQValues)

        # Return action distribution
        return softMaxQValues.numpy()[0]

    def get_graph_features(self, env, source, destination):
        """
        Obtain graph features for model
        """
        # We normalize the capacities
        capacity_feature = env.graph_state[:, 0] / env.maxCapacity

        sample = {
            'num_edges': env.numEdges,
            'length': env.firstTrueSize,
            'capacity': paddle.to_tensor(capacity_feature, dtype='float32'),
            'bw_allocated': paddle.to_tensor(env.bw_allocated_feature, dtype='float32'),
            'first': env.first,
            'second': env.second
        }

        sample['capacity'] = paddle.fluid.layers.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])

        # The hidden states of the links are composed of the link capacity and bw_allocated padded with zeros
        # Notice that the bw_allocated is stored as one-hot vector encoding to make it easier to learn for the GNN
        hiddenStates = paddle.fluid.layers.concat([sample['capacity'], sample['bw_allocated']], axis=1)
        paddings = paddle.fluid.initializer.Constant(
            [[0, 0], [0, self.actor_hparams['link_state_dim'] - 1 - self.actor_hparams['num_demands']]])
        link_state = paddle.vision.transforms.Pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                  'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs

    def update_actor_params(self, gradients):
        self.current_weights += gradients
        self.actor.set_weights(self.current_weights)

    def get_num_params(self):
        return self.actor.get_number_weights()[1]
        # return 6136


def obtain_ranking(vec):
    ranks = np.empty(vec.size, dtype=np.float32)
    ranks[vec.argsort()] = np.arange(vec.size, dtype=np.float32)
    return (ranks / (vec.size - 1)) - 0.5


def distribute_episodes(number_episodes, number_processes, heuristic_handicap=1):
    # Split the number of episodes between processes
    if number_episodes < number_processes:
        distributed_episodes = np.zeros(number_processes, dtype=int)
        distributed_episodes[-number_episodes:] = 1
        mpi_print("WARNING: less episodes than processes")
    else:
        distributed_episodes = np.empty(number_processes, dtype=int)
        min_episodes_per_episode = number_episodes // number_processes
        distributed_episodes[:] = min_episodes_per_episode

        remainder_episodes = number_episodes - (min_episodes_per_episode * number_processes)
        if 0 < remainder_episodes:
            distributed_episodes[:remainder_episodes] += 1

        # Heuristic: remove episodes from the main process to compensate for its additional responsibilities
        if heuristic_handicap and number_processes > 1:
            if distributed_episodes[0] < heuristic_handicap:
                heuristic_handicap = distributed_episodes[0]
                distributed_episodes[0] = 0
            else:
                distributed_episodes[0] -= heuristic_handicap
            # Start assigning them to the back first, as before we where handing it to the front
            min_episodes_per_episode = heuristic_handicap // (number_processes - 1)
            distributed_episodes[1:] += min_episodes_per_episode
            remainder_episodes = heuristic_handicap - (min_episodes_per_episode * (number_processes - 1))
            if 0 < remainder_episodes:
                distributed_episodes[-remainder_episodes:] += 1

    # Obtain the positions (accumulated indices)
    accumulated_episodes = np.zeros_like(distributed_episodes)
    for ii in range(1, number_processes):
        accumulated_episodes[ii] = accumulated_episodes[ii - 1] + distributed_episodes[ii - 1]

    # Obtain the positions to load the episodes
    loaded_episodes = np.zeros(number_processes + 1, dtype=int)
    loaded_episodes[:number_processes] = accumulated_episodes
    loaded_episodes[number_processes] = number_episodes

    return tuple(distributed_episodes), tuple(accumulated_episodes), loaded_episodes


def main_process():
    print("---main_process---")
    # Prepare optimizer
    optimizer = getattr(optimizers, HYPERPARAMS["optimizer"])(alpha=HYPERPARAMS["lr"])
    # Obtain the normalization factor for the gradient
    GRADIENT_FACTOR = 1 / (2 * HYPERPARAMS["number_mutations"] * HYPERPARAMS["param_noise_std"])

    # Name of training evaluations
    training_graph_names = "+".join(HYPERPARAMS["tr_graph_topologies"])
    rollout_time_buff = np.empty(1, dtype=np.float32)

    # Initial evaluation
    total_eval_rewards = np.empty((EVAL_LOADING_INDICES[-1], len(HYPERPARAMS["eval_graph_topologies"])),
                                  dtype=np.float32)
    # Perform evaluations
    local_eval_rewards = agent.perform_eval_rollouts(EVAL_EPISODES_PER_WORKER[mpi_rank], "eval_init.pckl")
    print("reward", np.mean(local_eval_rewards))
    # Gather results
    mpi_comm.Gatherv(local_eval_rewards, [total_eval_rewards, ADJ_EVAL_EPISODES_PER_WORKER, ADJ_EVAL_MPI_INDICES,
                                          MPI.FLOAT], 0)

    # Main loop
    for iters in range(HYPERPARAMS["episode_iterations"]):
        print("OTN ROUTING ({} Topology) GDES Iteration {}".format(training_graph_names, iters))

        local_perturbations, global_perturbations = perturbations_gen.global_obtain_perturbations(EPISODES_PER_WORKER,
                                                                                                  LOADING_INDICES,
                                                                                                  HYPERPARAMS[
                                                                                                      "number_mutations"])

        # Allocate memory for total rewards
        global_positive_rewards = np.empty(HYPERPARAMS["number_mutations"], dtype=np.float32)
        global_negative_rewards = np.empty_like(global_positive_rewards)

        # Perform own episodes (if necessary)
        local_positive_rewards, local_negative_rewards = agent.perform_tr_rollouts(local_perturbations, iters)

        mpi_comm.Reduce(np.zeros(1, dtype=np.float32), rollout_time_buff, op=MPI.MAX)
        mpi_comm.Gatherv(local_positive_rewards, [global_positive_rewards, EPISODES_PER_WORKER, MPI_INDICES, MPI.FLOAT])
        mpi_comm.Gatherv(local_negative_rewards, [global_negative_rewards, EPISODES_PER_WORKER, MPI_INDICES, MPI.FLOAT])

        # Perform reward scaling
        ranked_positive_rewards = obtain_ranking(global_positive_rewards)
        ranked_negative_rewards = obtain_ranking(global_negative_rewards)
        del global_positive_rewards, global_negative_rewards
        # obtain positive and negative rewards
        final_rewards = ranked_positive_rewards - ranked_negative_rewards
        del ranked_positive_rewards, ranked_negative_rewards
        assert final_rewards.size == global_perturbations.shape[0]

        # Obtain gradients
        # We need to invert it since we want to maximize, but the optimizer is meant for minimization
        gradient = -1 * np.dot(final_rewards, global_perturbations) * GRADIENT_FACTOR
        # We also consider the L2 regularization
        l2_regularization = agent.current_weights * HYPERPARAMS["l2_coeff"]
        # Obtain updated parameters
        parameters = optimizer.optimize(gradient + l2_regularization)
        mpi_comm.Bcast(parameters, root=0)
        agent.update_actor_params(parameters)
        del final_rewards, global_perturbations, l2_regularization, parameters

        if iters % HYPERPARAMS["evaluation_period"] == 0:
            print("OTN ROUTING ({} Topology) GDES Iteration {} -- Evaluation"
                  .format(training_graph_names, iters))
            # Perform evaluations
            local_eval_rewards = agent.perform_eval_rollouts(EVAL_EPISODES_PER_WORKER[mpi_rank],
                                                             "eval_{}.pckl".format(iters))
            print(iters, "reward:", np.mean(local_eval_rewards))
            save_reward('ES-' + str(graph_topology), iters, np.mean(local_eval_rewards))
            # Gather results
            mpi_comm.Gatherv(local_eval_rewards, [total_eval_rewards, ADJ_EVAL_EPISODES_PER_WORKER,
                                                  ADJ_EVAL_MPI_INDICES, MPI.FLOAT], 0)


def worker_process():
    # Initial evaluation
    # Perform evaluations
    print("---worker_process---")
    local_eval_rewards = agent.perform_eval_rollouts(EVAL_EPISODES_PER_WORKER[mpi_rank], "eval_init.pckl")
    # Gather results
    mpi_comm.Gatherv(local_eval_rewards, [NULL_BUFF, ADJ_EVAL_EPISODES_PER_WORKER, ADJ_EVAL_MPI_INDICES, MPI.FLOAT], 0)

    # Prepare time vars
    t1 = t2 = 0
    # Prepare parameters buffer
    parameters = np.empty(number_params, dtype=np.float32)

    # Main loop
    for iters in range(HYPERPARAMS["episode_iterations"]):

        # Sample random numbers
        local_perturbations = perturbations_gen.obtain_perturbations(EPISODES_PER_WORKER[mpi_rank])
        # Perform policy rollouts
        t1 = perf_counter()
        local_positive_rewards, local_negative_rewards = agent.perform_tr_rollouts(local_perturbations, iters)
        t2 = perf_counter()

        # Send numbers
        mpi_comm.Reduce(np.array([t2 - t1], dtype=np.float32), NULL_BUFF, op=MPI.MAX)
        mpi_comm.Gatherv(local_positive_rewards, [NULL_BUFF, EPISODES_PER_WORKER, MPI_INDICES,
                                                  MPI.FLOAT])
        mpi_comm.Gatherv(local_negative_rewards, [NULL_BUFF, EPISODES_PER_WORKER, MPI_INDICES,
                                                  MPI.FLOAT])

        # Update gradients
        mpi_comm.Bcast(parameters, root=0)
        agent.update_actor_params(parameters)

        # Perform evaluation
        if iters % HYPERPARAMS["evaluation_period"] == 0:
            # Perform evaluations
            local_eval_rewards = agent.perform_eval_rollouts(EVAL_EPISODES_PER_WORKER[mpi_rank],
                                                             "eval_{}.pckl".format(iters))
            # Gather results
            mpi_comm.Gatherv(local_eval_rewards, [NULL_BUFF, ADJ_EVAL_EPISODES_PER_WORKER, ADJ_EVAL_MPI_INDICES,
                                                  MPI.FLOAT], 0)


if __name__ == "__main__":
    mpi_print("script start")
    NULL_BUFF = np.empty(0, dtype=np.float32)

    # Load Configuration
    argsc = 'configs/ES/nsfnet.config'
    with open(argsc, 'r') as config_file:
        HYPERPARAMS = json.loads(config_file.read())

    # Get the environment and extract the number of actions.
    agent = ESPolicy(HYPERPARAMS["gnn-params"],
                     HYPERPARAMS["list_of_demands"],
                     HYPERPARAMS.get("params_file", None),
                     HYPERPARAMS["param_noise_std"],
                     HYPERPARAMS["action_noise_std"],
                     HYPERPARAMS["number_mutations"],
                     None)
    # Get the number of weights
    number_params = agent.get_num_params()

    # 由于镜像采样，突变的真实数量将增加一倍
    EPISODES_PER_WORKER, MPI_INDICES, LOADING_INDICES = distribute_episodes(HYPERPARAMS["number_mutations"], mpi_size)

    EVAL_EPISODES_PER_WORKER, EVAL_MPI_INDICES, EVAL_LOADING_INDICES = distribute_episodes(
        HYPERPARAMS["evaluation_episodes"], mpi_size, 0)
    # Adjusted loading constants for multiple-environment evaluation
    ADJ_EVAL_EPISODES_PER_WORKER = tuple(ii * len(HYPERPARAMS["eval_graph_topologies"])
                                         for ii in EVAL_EPISODES_PER_WORKER)
    ADJ_EVAL_MPI_INDICES = tuple(ii * len(HYPERPARAMS["eval_graph_topologies"]) for ii in EVAL_MPI_INDICES)

    # Sample random numbers
    perturbations_gen = PerturbationsGenerator(number_params)
    print("mpi_rnk:", mpi_rank)
    if mpi_rank:
        worker_process()
    else:
        main_process()

    mpi_comm.Barrier()
