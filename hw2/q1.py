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
parser=argparse.ArgumentParser()
parser.add_argument("--stages",type=int,default=10,choices=[10,20])
args=parser.parse_args()

def T(J,verbose=False,stage=None):
    '''
    Bellman operator for maximising reward

    Args:
    * verbose: Print policy each time T is operated.
    * stage: Which stage to operate T at.
    '''
    if verbose and stage:
        policy=np.argmax(np.sum(r*P+P*J.T[np.newaxis,:,np.newaxis],axis=1),axis=1)
        cost=np.amax(np.sum(r*P+P*J[np.newaxis,:,np.newaxis],axis=1),axis=1)
        print(f"Policy at stage {stage} is {policy}, cost at stage {cost}")
    else:
        return np.amax(np.sum(r*P+P*J[np.newaxis,:,np.newaxis],axis=1),axis=1)

def test_T():
    '''
    Sample test for T
    ''' 
    print (T(np.zeros(3)))

def read_optimal_policy(J_opt):
    '''
    Prints policy for a particular optimal J.

    Agrs:
    * J_opt: Optimal reward.
    '''
    actions= np.argmax(np.sum(r*P+P*J[np.newaxis,:,np.newaxis],axis=1),axis=1)
    return {f"state {state}":f"action {action}" for state,action in zip(range(3),actions)}

# State vector, start at stage N, zero end stage rewards.
J=np.zeros(3)

# P matrix
P=np.zeros((3,3,3))
P[:,:,0]=np.array([[1/2,1/4,1/4],[1/16,3/4,3/16],[1/4,1/8,5/8]])
P[:,:,1]=np.array([[1/2,0,1/2],[1/16,7/8,1/16],[0,0,0]])
P[:,:,2]=np.array([[1/4,1/4,1/2],[1/8,3/4,1/8],[3/4,1/16,3/16]])

# reward matrix
r=np.zeros((3,3,3))
r[:,:,0]=np.array([[10,4,8],[8,2,4],[4,6,4]])
r[:,:,1]=np.array([[14,0,18],[8,16,8],[0,0,0]])
r[:,:,2]=np.array([[10,2,8],[6,4,2],[4,0,8]])

print(f"Starting with end stage costs as {J}")
count=1
N=args.stages
while count < N:
    J=T(J)
    print(f"Values of J {J} at stage {N-count}")
    print(f"Optimal policy is at stage {N-count} is {read_optimal_policy(J)}")
    count+=1

print(f"Optimal policy is {read_optimal_policy(J)}")
