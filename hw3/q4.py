import numpy as np

P = np.zeros((2,2,2))
g = np.zeros((2,2,2))

P[:,:,0]= np.array([[0.5,0.5],[0.4,0.6]])
P[:,:,1]= np.array([[0.8,0.2],[0.7,0.3]])
g[:,:,0]= np.array([[-9,-3],[-3,7]])
g[:,:,1]= np.array([[-4,-4],[-1,19]])


def value_iter(P= P, g = g, alpha = 0.9, count = 10):

    print("----------------")
    print("Value Iteration")

    J= np.zeros(2)
    for i in range(count):
        policy=np.argmin(np.sum(g*P+alpha * P*J.T[np.newaxis,:,np.newaxis],axis=1),axis=1)
        cost=np.amin(np.sum(g*P+alpha * P * J[np.newaxis,:,np.newaxis],axis=1),axis=1)
        print(f"Minima over {np.sum(g*P+alpha * P * J[np.newaxis,:,np.newaxis],axis=1)}")
        J=cost
        print(f"Policy at stage {i+1} is {policy}, cost at stage {cost}")

    print("----------------")

def policy_iter(action= [0,0],P = P, g= g, alpha =0.9, count=3):

    print("----------------")
    print("Policy Iter")
    action=np.array(action)

    for i in range(count):
        P_=np.zeros((2,2))
        P_[0,:]=P[0,:,action[0]]
        P_[1,:]=P[1,:,action[1]]

        print(f"P Matrix {P_}")

        g_=np.zeros((2,2))
        g_[:,0]=P[:,0,action[0]]*g[:,0,action[0]]
        g_[:,1]=P[:,1,action[1]]* g[:,1,action[1]]

        g_=np.sum(g_,axis=1)

        print(f"G Matrix {g_}")

        J_=np.linalg.inv((np.eye(2)-alpha * P_))
        J_=np.dot(J_,g_)

        print(f"cost at stage {J_}")

        action=np.argmin(np.sum(g*P+alpha * P * J_.T[np.newaxis,:,np.newaxis],axis=1),axis=1)

        print("Verbosity", np.sum(g*P+alpha * P * J_.T[np.newaxis,:,np.newaxis],axis=1))

        print(f"Policy at stage {i+1} is {action}")

    print("----------------")


def invert(action,P=P,g=g,alpha=0.9):

    P_=np.zeros((2,2))
    P_[0,:]=P[0,:,action[0]]
    P_[1,:]=P[1,:,action[1]]

    print(f"P Matrix {P_}")

    g_=np.zeros((2,2))
    g_[0,:]=P[0,:,action[0]]*g[0,:,action[0]]
    g_[1,:]=P[1,:,action[1]]* g[1,:,action[1]]

    g_=np.sum(g_,axis=1)
    g_=g_.T

    print(f"G Matrix {g_}")

    print(g_)

    J_=np.linalg.inv((np.eye(2)-alpha * P_))
    J_=np.dot(J_,g_)

    print (f"Value for {action} is {J_}")


action_list=[[i,j] for i in range(2) for j in range(2)]

for action in action_list:
    invert(action)

policy_iter()

value_iter(count=4)

policy_iter(alpha=0.1)
