import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_likelihood(matrix, xlabels=list(range(9)), ylabels=list(range(9)), title_str="Likelihood distribution (A)"):
    """
    Plots a 2-D likelihood matrix as a heatmap
    """

    if not np.isclose(matrix.sum(axis=0), 1.0).all():
        raise ValueError(
            "Distribution not column-normalized! Please normalize (ensure matrix.sum(axis=0) == 1.0 for all columns)")

    fig = plt.figure(figsize=(6, 6))
    ax = sns.heatmap(matrix, xticklabels=xlabels, yticklabels=ylabels, cmap='gray', cbar=False, vmin=0.0, vmax=1.0)
    plt.title(title_str)
    plt.show()


def plot_grid(grid_locations, num_x=3, num_y=3):
    """
    Plots the spatial coordinates of GridWorld as a heatmap, with each (X, Y) coordinate
    labeled with its linear index (its `state id`)
    """

    grid_heatmap = np.zeros((num_x, num_y))
    for linear_idx, location in enumerate(grid_locations):
        y, x = location
        grid_heatmap[y, x] = linear_idx
    sns.set(font_scale=1.5)
    sns.heatmap(grid_heatmap, annot=True, cbar=False, fmt='.0f', cmap='crest')
    plt.show()


def plot_point_on_grid(state_vector, grid_locations):
    """
    Plots the current location of the agent on the grid world
    """
    state_index = np.where(state_vector)[0][0]
    y, x = grid_locations[state_index]
    grid_heatmap = np.zeros((3, 3))
    grid_heatmap[y, x] = 1.0
    sns.heatmap(grid_heatmap, cbar=False, fmt='.0f')
    plt.show()


def plot_beliefs(belief_dist, title_str=""):
    """
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    """

    if not np.isclose(belief_dist.sum(), 1.0):
        raise ValueError("Distribution not normalized! Please normalize")

    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title_str)
    plt.show()


from pymdp import utils

my_categorical = np.random.rand(3)
# my_categorical = np.array([0.5, 0.3, 0.8]) # could also just write in your own numbers
my_categorical = utils.norm_dist(my_categorical)  # normalizes the distribution so it integrates to 1.0

print(my_categorical.reshape(-1, 1))  # we reshape it to display it like a column vector
print(f'Integral of the distribution: {round(my_categorical.sum(), 2)}')

sampled_outcome = utils.sample(my_categorical)
print(f'Sampled outcome: {sampled_outcome}')
plot_beliefs(my_categorical, title_str="A random (unconditional) Categorical distribution")

# initialize it with random numbers
p_x_given_y = np.random.rand(3, 4)
print(p_x_given_y.round(3))
# normalize it: normalize first dim by default
p_x_given_y = utils.norm_dist(p_x_given_y)
print(p_x_given_y.round(3))
print(p_x_given_y[:, 0].reshape(-1, 1))
print(f'Integral of P(X|Y=0): {p_x_given_y[:, 0].sum()}')

""" Create a P(Y) and P(X|Y) using the same numbers from the slides """
p_y = np.array([0.75, 0.25])  # this is already normalized - you don't need to `utils.norm_dist()` it!
# the columns here are already normalized - you don't need to `utils.norm_dist()` it!
p_x_given_y = np.array([[0.6, 0.5],
                        [0.15, 0.41],
                        [0.25, 0.09]])
print(p_y.round(3).reshape(-1, 1))
print(p_x_given_y.round(3))
""" Calculate the expectation using numpy's dot product functionality """
# first version of the dot product (using the method of a numpy array)
E_x_wrt_y = p_x_given_y.dot(p_y)
# second version of the dot product (using the function np.dot with two arguments)
# E_x_wrt_y = np.dot(p_x_given_y, p_y)
print(E_x_wrt_y)
print(f'Integral: {E_x_wrt_y.sum().round(3)}')

import itertools

""" Create  the grid locations in the form of a list of (Y, X) tuples -- HINT: use itertools """
grid_locations = list(itertools.product(range(3), repeat=2))
print(grid_locations)
plot_grid(grid_locations)

""" Create variables for the storing the dimensionalities of the hidden states and the observations """
n_states = len(grid_locations)
n_observations = len(grid_locations)
print(f'Dimensionality of hidden states: {n_states}')
print(f'Dimensionality of observations: {n_observations}')
A = np.zeros((n_observations, n_states))
np.fill_diagonal(A, 1.0)
plot_likelihood(A, title_str="A matrix or $P(o|s)$")
A_noisy = A.copy()
# this line says: the probability of seeing yourself in location 0, given you're in location 0, is 1/3, AKA P(o == 0 | s == 0) = 0.3333....
A_noisy[0, 0] = 1 / 3.0  # corresponds to location (0,0)
# this line says: the probability of seeing yourself in location 1, given you're in location 0, is 1/3, AKA P(o == 1 | s == 0) = 0.3333....
A_noisy[1, 0] = 1 / 3.0  # corresponds to one step to the right from (0, 1)
# this line says: the probability of seeing yourself in location 3, given you're in location 0, is 1/3, AKA P(o == 3 | s == 0) = 0.3333....
A_noisy[3, 0] = 1 / 3.0  # corresponds to one step down from (1, 0)
plot_likelihood(A_noisy, title_str='modified A matrix where location (0,0) is "blurry"')
my_A_noisy = A_noisy.copy()

# locations 3 and 7 are the nearest neighbours to location 6
my_A_noisy[3, 6] = 1.0 / 3.0
my_A_noisy[6, 6] = 1.0 / 3.0
my_A_noisy[7, 6] = 1.0 / 3.0
plot_likelihood(my_A_noisy, title_str="Noisy A matrix now with TWO ambiguous locations")

actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


def create_B_matrix():
    B = np.zeros((len(grid_locations), len(grid_locations), len(actions)))
    for action_id, action_label in enumerate(actions):
        for curr_state, grid_location in enumerate(grid_locations):
            y, x = grid_location
            if action_label == "UP":
                next_y = y - 1 if y > 0 else y
                next_x = x
            elif action_label == "DOWN":
                next_y = y + 1 if y < 2 else y
                next_x = x
            elif action_label == "LEFT":
                next_x = x - 1 if x > 0 else x
                next_y = y
            elif action_label == "RIGHT":
                next_x = x + 1 if x < 2 else x
                next_y = y
            elif action_label == "STAY":
                next_x = x
                next_y = y
            new_location = (next_y, next_x)
            next_state = grid_locations.index(new_location)
            B[next_state, curr_state, action_id] = 1.0
    return B


B = create_B_matrix()
""" Define a starting location"""
starting_location = (1, 0)
"""get the linear index of the state"""
state_index = grid_locations.index(starting_location)
"""  and create a state vector out of it """
starting_state = utils.onehot(state_index, n_states)
plot_point_on_grid(starting_state, grid_locations)
plot_beliefs(starting_state, "Categorical distribution over the starting state")
""" Generate the next state vector, given the starting state and the B matrix"""
right_action_idx = actions.index("RIGHT")
next_state = B[:, :, right_action_idx].dot(starting_state)  # input the indices to the B matrix
""" Plot the next state, after taking the action """
plot_point_on_grid(next_state, grid_locations)
prev_state = next_state.copy()
down_action_index = actions.index("DOWN")
next_state = B[:, :, down_action_index].dot(prev_state)
"""  Plot the new state vector, after making the movement """
plot_point_on_grid(next_state, grid_locations)

""" Create an empty vector to store the preferences over observations """
C = np.zeros(n_observations)
""" Choose an observation index to be the 'desired' rewarding index, and fill out the C vector accordingly """
desired_location = (2, 2)  # choose a desired location
desired_location_index = grid_locations.index(
    desired_location)  # get the linear index of the grid location, in terms of 0 through 8
C[desired_location_index] = 1.0  # set the preference for that location to be 100%, i.e. 1.0
"""  Let's look at the prior preference distribution """
plot_beliefs(C, title_str="Preferences over observations")

""" Create a D vector, basically a belief that the agent has about its own starting location """

# create a one-hot / certain belief about initial state
D = utils.onehot(0, n_states)
# demonstrate hwo belief about initial state can also be uncertain / spread among different possible initial states
# alternative, where you have a degenerate/noisy prior belief
# D = utils.norm_dist(np.ones(n_states))
""" Let's look at the prior over hidden states """
plot_beliefs(D, title_str="Prior beliefs over states")

from pymdp.maths import softmax
from pymdp.maths import spm_log_single as log_stable


def infer_states(observation_index, A, prior):
    log_likelihood = log_stable(A[observation_index, :])
    log_prior = log_stable(prior)
    qs = softmax(log_likelihood + log_prior)
    return qs


qs_past = utils.onehot(4, n_states)  # agent believes they were at location 4 -- i.e. (1,1) one timestep ago
last_action = "UP"  # the agent knew it moved "UP" one timestep ago
action_id = actions.index(last_action)  # get the action index for moving "UP"
prior = B[:, :, action_id].dot(qs_past)
observation_index = 1
qs_new = infer_states(observation_index, A, prior)
plot_beliefs(qs_new, title_str="Beliefs about hidden states")

observation_index = 2  # this is like the agent is seeing itself in location (0, 2)
qs_new = infer_states(observation_index, A, prior)
plot_beliefs(qs_new)

A_partially_ambiguous = softmax(A)
print(A_partially_ambiguous.round(3))

noisy_prior = softmax(prior)
plot_beliefs(noisy_prior)

qs_new = infer_states(observation_index, A_partially_ambiguous, noisy_prior)
plot_beliefs(qs_new)


def get_expected_states(B, qs_current, action):
    qs_u = B[:, :, action].dot(qs_current)
    return qs_u


def get_expected_observations(A, qs_u):
    qo_u = A.dot(qs_u)
    return qo_u


def entropy(A):
    H_A = - (A * log_stable(A)).sum(axis=0)
    return H_A


def kl_divergence(qo_u, C):
    return (log_stable(qo_u) - log_stable(C)).dot(qo_u)


state_idx = grid_locations.index((1, 1))
state_vector = utils.onehot(state_idx, n_states)
plot_point_on_grid(state_vector, grid_locations)
qs_current = state_vector.copy()
plot_beliefs(qs_current, title_str="Where do we believe we are?")
plot_grid(grid_locations)
desired_idx = grid_locations.index((1, 2))
C = utils.onehot(desired_idx, n_observations)
plot_beliefs(C, title_str="Preferences")
left_idx = actions.index("LEFT")
right_idx = actions.index("RIGHT")
print(f'Action index of moving left: {left_idx}')
print(f'Action index of moving right: {right_idx}')

G = np.zeros(2)  # store the expected free energies for each action in here

qs_u_left = get_expected_states(B, qs_current, left_idx)
H_A = entropy(A)
qo_u_left = get_expected_observations(A, qs_u_left)
predicted_uncertainty_left = H_A.dot(qs_u_left)
predicted_divergence_left = kl_divergence(qo_u_left, C)
G[0] = predicted_uncertainty_left + predicted_divergence_left

qs_u_right = get_expected_states(B, qs_current, right_idx)
H_A = entropy(A)
qo_u_right = get_expected_observations(A, qs_u_right)
predicted_uncertainty_right = H_A.dot(qs_u_right)
predicted_divergence_right = kl_divergence(qo_u_right, C)
G[1] = predicted_uncertainty_right + predicted_divergence_right

print(f'Expected free energy of moving left: {G[0]}\n')
print(f'Expected free energy of moving right: {G[1]}\n')
Q_u = softmax(-G)
print(f'Probability of moving left: {Q_u[0]}')
print(f'Probability of moving right: {Q_u[1]}')


def calculate_G(A, B, C, qs_current, actions):
    G = np.zeros(len(actions))  # vector of expected free energies, one per action
    H_A = entropy(A)  # entropy of the observation model, P(o|s)
    for action_i in range(len(actions)):
        qs_u = get_expected_states(B, qs_current, action_i)
        qo_u = get_expected_observations(A, qs_u)
        pred_uncertainty = H_A.dot(qs_u)  # predicted uncertainty, i.e. expected entropy of the A matrix
        pred_div = kl_divergence(qo_u, C)  # predicted divergence
        G[action_i] = pred_uncertainty + pred_div  # sum them together to get expected free energy
    return G


# complete guide
class GridWorldEnv():

    def __init__(self, starting_state=(0, 0)):
        self.init_state = starting_state
        self.current_state = self.init_state
        print(f'Starting state is {starting_state}')

    def step(self, action_label):
        (Y, X) = self.current_state
        if action_label == "UP":
            Y_new = Y - 1 if Y > 0 else Y
            X_new = X
        elif action_label == "DOWN":
            Y_new = Y + 1 if Y < 2 else Y
            X_new = X
        elif action_label == "LEFT":
            Y_new = Y
            X_new = X - 1 if X > 0 else X
        elif action_label == "RIGHT":
            Y_new = Y
            X_new = X + 1 if X < 2 else X
        elif action_label == "STAY":
            Y_new, X_new = Y, X
        self.current_state = (Y_new, X_new)  # store the new grid location
        obs = self.current_state  # agent always directly observes the grid location they're in
        return obs

    def reset(self):
        self.current_state = self.init_state
        print(f'Re-initialized location to {self.init_state}')
        obs = self.current_state
        print(f'..and sampled observation {obs}')
        return obs


env = GridWorldEnv()
A = np.eye(n_observations, n_states)
B = create_B_matrix()
C = utils.onehot(grid_locations.index((2, 2)), n_observations)
D = utils.onehot(grid_locations.index((1, 2)), n_states)
actions = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

env = GridWorldEnv(starting_state=(1, 2))


def run_active_inference_loop(A, B, C, D, actions, env, T=5):
    prior = D.copy()  # initial prior should be the D vector
    obs = env.reset()  # initialize the `obs` variable to be the first observation you sample from the environment, before `step`-ing it.
    for t in range(T):
        print(f'Time {t}: Agent observes itself in location: {obs}')
        # convert the observation into the agent's observational state space (in terms of 0 through 8)
        obs_idx = grid_locations.index(obs)
        # perform inference over hidden states
        qs_current = infer_states(obs_idx, A, prior)
        plot_beliefs(qs_current, title_str=f"Beliefs about location at time {t}")
        # calculate expected free energy of actions
        G = calculate_G(A, B, C, qs_current, actions)
        # compute action posterior
        Q_u = softmax(-G)
        # sample action from probability distribution over actions
        chosen_action = utils.sample(Q_u)
        # compute prior for next timestep of inference
        prior = B[:, :, chosen_action].dot(qs_current)
        # update generative process
        action_label = actions[chosen_action]
        obs = env.step(action_label)
    return qs_current


# qs = run_active_inference_loop(A, B, C, D, actions, env, T=5)


D = utils.onehot(grid_locations.index((0, 0)), n_states)  # let's have the agent believe it starts in location (0,0)
env = GridWorldEnv(starting_state=(0, 0))
# qs = run_active_inference_loop(A, B, C, D, actions, env, T = 5)

from pymdp.control import construct_policies

policy_len = 4
n_actions = len(actions)
# we have to wrap `n_states` and `n_actions` in a list for reasons that will become clear in Part II
all_policies = construct_policies([n_states], [n_actions], policy_len=policy_len)

print(f'Total number of policies for {n_actions} possible actions and a planning horizon of {policy_len}: {len(all_policies)}')


def calculate_G_policies(A, B, C, qs_current, policies):
    G = np.zeros(len(policies))  # initialize the vector of expected free energies, one per policy
    H_A = entropy(A)  # can calculate the entropy of the A matrix beforehand, since it'll be the same for all policies
    for policy_id, policy in enumerate(policies):
        t_horizon = policy.shape[0]  # temporal depth of the policy
        G_pi = 0.0  # initialize expected free energy for this policy
        for t in range(t_horizon):  # loop over temporal depth of the policy
            action = policy[t, 0]  # action entailed by this particular policy, at time `t`
            if t == 0:
                qs_prev = qs_current
            else:
                qs_prev = qs_pi_t
            qs_pi_t = get_expected_states(B, qs_prev, action)
            qo_pi_t = get_expected_observations(A, qs_pi_t)
            kld = kl_divergence(qo_pi_t, C)
            G_pi_t = H_A.dot(qs_pi_t) + kld
            G_pi += G_pi_t
        G[policy_id] += G_pi
    return G


def compute_prob_actions(actions, policies, Q_pi):
    P_u = np.zeros(len(actions))  # initialize the vector of probabilities of each action
    for policy_id, policy in enumerate(policies):
        P_u[int(policy[0, 0])] += Q_pi[policy_id]  # get the marginal probability for the given action, entailed by this policy at the first timestep
    P_u = utils.norm_dist(P_u)  # normalize the action probabilities
    return P_u


def active_inference_with_planning(A, B, C, D, n_actions, env, policy_len=2, T=5):
    """ Initialize prior, first observation, and policies """
    prior = D  # initial prior should be the D vector
    obs = env.reset()  # get the initial observation
    policies = construct_policies([n_states], [n_actions], policy_len=policy_len)
    for t in range(T):
        print(f'Time {t}: Agent observes itself in location: {obs}')
        # convert the observation into the agent's observational state space (in terms of 0 through 8)
        obs_idx = grid_locations.index(obs)
        # perform inference over hidden states
        qs_current = infer_states(obs_idx, A, prior)
        plot_beliefs(qs_current, title_str=f"Beliefs about location at time {t}")
        # calculate expected free energy of actions
        G = calculate_G_policies(A, B, C, qs_current, policies)
        # to get action posterior, we marginalize P(u|pi) with the probabilities of each policy Q(pi), given by \sigma(-G)
        Q_pi = softmax(-G)
        # compute the probability of each action
        P_u = compute_prob_actions(actions, policies, Q_pi)
        # sample action from probability distribution over actions
        chosen_action = utils.sample(P_u)
        # compute prior for next timestep of inference
        prior = B[:, :, chosen_action].dot(qs_current)
        # step the generative process and get new observation
        action_label = actions[chosen_action]
        obs = env.step(action_label)
    return qs_current


D = utils.onehot(grid_locations.index((0,0)), n_states) # let's have the agent believe it starts in location (0,0) (upper left corner)
env = GridWorldEnv(starting_state = (0,0))
qs_final = active_inference_with_planning(A, B, C, D, n_actions, env, policy_len = 3, T = 10)