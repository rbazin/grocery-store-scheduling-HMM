from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import norm
import numpy as np
from time import time
import argparse
import copy

# Argument parsing


def get_args():
    parser = argparse.ArgumentParser(
        description="Computes value iteration or policy iteration algorithms for the grocery store scheduling problem"
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="V",
        help="Defines the algorithm used, can be V for value iteration or P for policy iteration",
    )
    parser.add_argument(
        "--N", type=int, default=10, help="Maximum number of customers in store"
    )
    parser.add_argument(
        "--M", type=int, default=20, help="Maximum amount of boxes of mangoes in store"
    )
    parser.add_argument(
        "--E1", type=int, default=10, help="Fixed cost of ordering boxes of mangoes"
    )
    parser.add_argument(
        "--E2",
        type=int,
        default=1,
        help="Proportional cost of ordering boxes of mangoes",
    )
    parser.add_argument(
        "--E3",
        type=int,
        default=1,
        help="Cost of maintaining boxes of mangoes in good shape",
    )
    parser.add_argument(
        "--S", type=int, default=3, help="Selling price of boxes of mangoes"
    )
    parser.add_argument(
        "--C", type=int, default=1, help="Cost of unsatisfied customers"
    )
    parser.add_argument(
        "--P",
        type=float,
        default=0.3,
        help="Probability that a customer takes a box of mangoes",
    )
    parser.add_argument(
        "--T", type=int, default=24, help="Hour at which the store closes"
    )
    parser.add_argument(
        "--customer_law",
        type=str,
        default="poisson",
        help="Customer distribution law, can be either 'poisson' or 'normal'",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.9,
        help="Discount factor",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Max number of iterations to compute the algorithms on",
    )
    return parser.parse_args()


def get_states(M, T):
    states = []
    for t in range(0, T + 1):
        for x in range(0, M + 1):
            states.append((x, t))
    return states


def get_actions(M):
    return list(range(0, M + 1))


def average_reward(state, action):
    X, t = state
    res = sum(
        reward_function(state, k, action)
        * (
            sum(
                binom.pmf(k, n, P) * P_nbr_clients(CLIENT_LAW, n, t + 1)
                for n in range(0, N + 1)
            )
        )
        for k in range(0, M + 1)
    )
    return res


def reward_function(state, K, action):
    X, t = state
    R = 0
    if action > 0:
        R += -E1
    R += -E2 * action
    R += S * np.min((K, X))
    if X > K:
        R += -E3 * (X - K)
    else:
        R += -C * (K - X)
    return R


def g(t):
    return 1 - (t - 12) ** 2 / 144


def P_nbr_clients(law, nbr, time):

    if law == "poisson":
        L = 9 / S * g(time)
        return poisson.pmf(nbr, L)

    M = 9 / S * g(time)
    sigma = 3 / S
    return norm.pdf(nbr, M, sigma)


def transition_model(k, state, action):

    X, t = state
    t_next = t + 1
    res = sum(
        binom.pmf(k, n, P) * P_nbr_clients(CLIENT_LAW, n, t_next)
        for n in range(0, N + 1)
    )
    return res


def value_iteration(states, actions):
    U = np.zeros((M + 1, T + 1))
    i = 0

    print("Starting value iteration")

    while True:
        U_old = U.copy()

        print(f"Iteration : {i}")

        for state in states:
            start = time()
            print(state)
            Q = []
            for action in actions:
                Q.append(
                    average_reward(state, action)
                    + GAMMA
                    * sum(
                        transition_model(k, state, action)
                        * U_old[np.min((state[0] + action - k, M)), state[1] + 1]
                        for k in range(0, state[0] + 1)
                    )
                )
            end = time()
            print(f"Time to compute utility for state {state} : {end - start:.2f}s")

            U[state] = max(Q)
            print(f"Utility for state {state} : {U[state]}")

        i += 1

        if all(U_old[state] == U[state] for state in states) or i > MAX_ITER:
            print("Value iteration stopped after", i, "iterations")
            break

    return U


def policy_evaluation(policy, states):
    U = np.zeros((M + 1, T + 1))
    j = 0

    while True:
        U_old = U.copy()

        for state in states:
            action = policy[state]
            U[state] = average_reward(state, action) + GAMMA * sum(
                transition_model(k, state, action)
                * U_old[np.min((state[0] + action - k, M)), state[1] + 1]
                for k in range(0, state[0] + 1)
            )
        j += 1

        if all(U_old[state] == U[state] for state in states) or j > MAX_ITER:
            break

    return U


def policy_improvement(U, states, actions):
    policy = {state: actions[0] for state in states}

    for state in states:
        Q = {}
        for action in actions:
            Q[action] = average_reward(state, action) + GAMMA * sum(
                transition_model(k, state, action)
                * U[np.min((state[0] + action - k, M)), state[1] + 1]
                for k in range(0, state[0] + 1)
            )
        policy[state] = max(Q, key=Q.get)

    return policy


def policy_iteration(states, actions):
    policy = {state: actions[0] for state in states}

    i = 0
    while True:
        policy_old = copy.deepcopy()
        U = policy_evaluation(policy, states)
        policy = policy_improvement(U, states, actions)

        if all(policy_old[state] == policy[state] for state in states) or i > MAX_ITER:
            break

    return policy


if __name__ == "__main__":
    # Parameters
    args = get_args()
    algo = args.algo
    N = args.N
    M = args.M
    E1 = args.E1
    E2 = args.E2
    E3 = args.E3
    S = args.S
    C = args.C
    P = args.P
    T = args.T
    GAMMA = args.gamma
    CLIENT_LAW = args.customer_law
    MAX_ITER = args.max_iter

    assert algo == "V" or algo == "P", "Please select a valid algorithm"
    assert (
        CLIENT_LAW == "poisson" or CLIENT_LAW == "normal"
    ), "Please select a valid customer law"

    states = get_states(M, T)
    actions = get_actions(M)

    if algo == "V":
        U = value_iteration(states, actions)
        print(U)
    else:
        policy = policy_iteration(states, actions)
        print("Optimal policy found :", policy)
        number_boxes = int(input("Choose the current number of boxes of mangoes\n"))
        time = int(input("Choose the time of the day"))
        state = (number_boxes, time)
        print(f"Your state is {number_boxes, time}")
        print(
            "best action in this state is to order {policy[state]} boxes, according to the optimal policy found"
        )
