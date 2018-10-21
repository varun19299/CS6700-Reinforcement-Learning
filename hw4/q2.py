'''
RL Assignment - 2
Question 1

Example Usage:
python3 q2.py --stages 25

Args:
* stages: Which stage to compute to, one of 10,20.
'''

# Library imports
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--stages", type=int, default=20, choices=[10, 20, 30])
parser.add_argument(
    "--iter_type",
    default="policy",
    choices=[
        "value",
        "policy",
        "mpi",
        "gauss"])
args = parser.parse_args()


def T(J, verbose=False, stage=None, alpha=0.9):
    '''
    Bellman operator for maximising reward

    Args:
    * verbose: Print policy each time T is operated.
    * stage: Which stage to operate T at.
    '''
    J_sub = np.ones((3,3,3))

    for i in range(3):
        J_sub[:,i,:] = J[i]
    prod_av=P * J_sub

    if verbose and stage:
        policy = np.argmax(
            np.sum(r * P + alpha * prod_av, axis=1), axis=1)
        cost = np.amax(np.sum(r * P + alpha * prod_av, axis=1), axis=1)
        print(f"Policy at stage {stage} is {policy}, cost at stage {cost}")
    else:
        return np.amax(np.sum(r * P + alpha * prod_av, axis=1), axis=1)


def policy_iter(actions, alpha=0.9, verbose=False):
    # Helper vars
    actions_array = []
    J_array = []
    counter = 0

    i = list(range(3))

    while True:
        # Policy evaluation
        actions_array.append(actions)
        div = np.eye(3) - alpha * P[i, :, actions[i]]
        div = np.linalg.inv(div)

        J = np.sum(r[i, :, actions[i]] *
                   P[i, :, actions[i]], axis=1)
        J = np.dot(div, J)

        J_array.append(J)

        if verbose:
            print(f"J value {J}")

        # Policy improvement
        actions_new = read_optimal_policy(J, alpha=alpha)

        # if np.max(np.abs(actions_new - actions)) == 0:
        if np.sum(actions_new != actions) == 0:
            if verbose:
                print(f"Counter value {counter}  \n\n")
            actions = read_optimal_policy(J, alpha = alpha,verbose=verbose)
            break
        else:
            actions = actions_new
            counter += 1

    return J_array, actions_array


def modified_policy_iter(actions, m_k=5, alpha=0.9, verbose=False):
    # Helper vars
    actions_array = []
    J_array = []
    counter = 0

    i = list(range(3))

    while True:
        # Policy evaluation
        actions_array.append(actions)
        # div = np.eye(3) - alpha *P[i, :, actions[i]]
        # div = np.linalg.inv(div)

        # J = np.sum(r[i, :, actions[i]] *
        #             P[i, :, actions[i]], axis=1)
        # J = np.dot(div, J)

        # Perform value iteration m_k times
        J = np.zeros(3)
        for i in range(m_k):
            J = T(J, alpha=alpha)

        J_array.append(J)

        if verbose:
            print(f"J value {J}")

        # Policy improvement
        actions_new = read_optimal_policy(J, alpha=alpha, verbose=verbose)

        if np.max(np.abs(actions_new - actions)) == 0:
            if verbose:
                print(f"Counter value {counter}  \n\n")
            actions = read_optimal_policy(J, alpha=alpha, verbose=verbose)
            break
        else:
            actions = actions_new
            counter += 1

    return J_array, actions_array


def test_T():
    '''
    Sample test for T
    '''
    print(T(np.zeros(3)))

def read_optimal_policy(J, alpha=0.9, verbose=False):
    '''
    Prints policy for a particular optimal J.

    Agrs:
    * J_opt: Optimal reward.
    '''
    J_sub = np.ones((3,3,3))

    for i in range(3):
        J_sub[:,i,:] = J[i]
    prod_av=P * J_sub
    # prod_av = np.zeros((3,3,3))

    # for i in range(3):
    #     for j in range(3):
    #         prod_av[i,:,j] = J[:] * P[i,:,j]

    actions = np.argmax(
        np.sum(r * P + alpha * prod_av , axis=1), axis=1)

    # Non vectorised version
    
    # actions = np.zeros(3, dtype = int)

    # for i in range(3):
    #     res = np.zeros(3)
    #     for j in range(3):
    #         res[j] = r[i,:,j]*P[i,:,j] + alpha * J * P[i,:,:] 
    #     res = np.sum(res,axis = 0)
    #     actions[i] = np.argmin(res)
    if verbose:
        print({
            f"state {state}": f"action {action}" for state,
            action in zip(
                range(3),
                actions)})
    return actions

def gauss_siedel(J, alpha=0.9, verbose=True, epi=1e-6):
    actions_array = []
    J_array = []

    count = 0

    while True:

        # Randomly update a state
        r = np.random.randint(3)
        up = T(J, alpha=alpha)[r]
        diff = J[r] - up
        J[r] = up

        if verbose:
            print(f"Count {count}")
            print(f"Values of J {J} at stage {count}")
            print(f"Optimal policy is at stage {count} is")
            actions_array.append(
                read_optimal_policy(
                    J, verbose=verbose, alpha=alpha))
        else:
            actions_array.append(
                read_optimal_policy(
                    J, verbose=verbose, alpha=alpha))

        if np.abs(diff) < epi:
            break

        else:
            J_array.append(J)
            count += 1

    print(f"Iterations needed : \n{count}")
    print("Optimal policy is :")
    read_optimal_policy(J, alpha=alpha, verbose=True)
    print(f"Cost is : \n{J}")

    return J_array, actions_array

# Define matrices such that indices are i,j, actions
# P matrix
P = np.zeros((3, 3, 3))

P[:, :, 0] = np.array(
    [[1 / 2, 1 / 4, 1 / 4], [1 / 2, 0, 1/2], [1 / 4, 1 / 4, 1/2]])
P[:, :, 1] = np.array([[1 / 16, 3/4, 3 / 16], [1 / 8, 3/4, 1 / 16], [1/8, 3/4, 1/8]])
P[:, :, 2] = np.array(
    [[1 / 4, 1 / 8, 5 / 8], [0,0,0], [3 / 4, 1 / 16, 3 / 16]])

# reward matrix
r = np.zeros((3, 3, 3))
r[:, :, 0] = np.array([[10, 4, 8], [14, 0, 18],[10, 2, 8]])
r[:, :, 1] = np.array([[8,2,4], [8, 16, 8], [6,4,2]])
r[:, :, 2] = np.array([[4,6,4], [0,0,0], [4,0,8]])

m_k = 10

for a in range(0, 100, 10):
    alpha = a / 100
    verbose = False
    print(f"BETA {alpha}")

    if args.iter_type == "value":
        # Value iteration
        J = np.zeros(3)

        print(f" Value Iteration: Starting with end stage costs as {J}")
        count = 1
        N = args.stages
        temp= T(J, alpha= alpha)
        while count < N or np.max(np.abs(J- temp)) > 1e-3:
            J = temp
            temp = T(J, alpha= alpha)
            if verbose:
                print(f"Values of J {J} at stage {N-count}")
                print(f"Optimal policy is at stage {N-count} is")
                read_optimal_policy(J, alpha=alpha)
            count += 1

        print("Optimal policy is ")
        read_optimal_policy(J, alpha=alpha, verbose=True)
        print(f"Cost is {J}")

    elif args.iter_type == "policy":
        # Policy iteration
        actions = np.zeros(3, dtype=int)
        print(f"Policy Iteration: Starting with initial policy {actions}")

        J_array, actions_array = policy_iter(
            actions, alpha=alpha, verbose=True)
        print("Final optimal policy")
        print({
            f"state {state}": f"action {action}" for state,
            action in zip(
                range(3),
                actions_array[-1])})
        print("Final optimal cost")
        print(J_array[-1])

    elif args.iter_type == "mpi":
        # Modified Policy iteration
        actions = np.zeros(3, dtype=int)
        print(
            f"Modified Policy Iteration: Starting with initial policy {actions}")

        J_array, actions_array = modified_policy_iter(
            actions, m_k=m_k, alpha=alpha, verbose=False)
        print("Final optimal policy")
        print({
            f"state {state}": f"action {action}" for state,
            action in zip(
                range(3),
                actions_array[-1])})
        print("Final MPI cost")
        print(J_array[-1])

        print("Final Optimal cost associated with the policy")
        J_array, _ = policy_iter(actions_array[-1], alpha=alpha, verbose=False)
        print(J_array[-1])

    elif args.iter_type == "gauss":
        # Gauss Siedel
        J = np.zeros(3)
        print(
            f"\nGauss Seidel Iteration: \n\nStarting with end stage costs as {J}")
        J_array, actions_array = gauss_siedel(J, alpha=alpha, verbose=False)

    print("\n\n ------------ \n\n")

