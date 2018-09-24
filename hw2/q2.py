'''
RL Assignment - 2
Question 2

Example Usage:
python3 q2.py --terminal 100 --stages 25

Args:
* terminal: Terminal state.
* stages: number of stages to apply DP for.
'''
# Library imports
import numpy as np 
import matplotlib.pyplot as plt
import argparse
import os

# Spice up colours
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Module imports
from bellman import *

# Parse agrs
parser=argparse.ArgumentParser()
parser.add_argument("--stages",type=int,default=10,choices=[10,25,-1],\
help="Number of stages to solve for DP, -1 indicates till convergence of reward.")
parser.add_argument("--terminal",type=int,default=99,choices=[99,3],\
help="Terminal State in the grid world.")
parser.add_argument("--supress",type=int,default=0,choices=[0,1],\
help="Whether to supress plots from showing up.")
args=parser.parse_args()

# We linearise states, go from 0 to 99.
terminal=args.terminal
bel=Bellman(terminal,np.arange(100),np.arange(4),args.stages,False)

bel.J=np.zeros((100))

{0: "Up", 1: "Down",2: "left", 3:"right"}
P=np.zeros((100,100,4))

for i in range(100):
    # Up is +10
    if (i<90):
        P[i,i+10,0]=0.8
    else:
        P[i,i,0]=0.8

    if i%10<9:
        P[i,i+1,0]=0.1
    else:
        P[i,i,0]=0.1
    if i%10>0:
        P[i,i-1,0]=0.1
    else:
        P[i,i,0]=0.1

    # Down is -10
    if (i>9):
        P[i,i-10,1]=0.8
    else:
        P[i,i,1]=0.8

    if i%10<9:
        P[i,i+1,1]=0.1
    else:
        P[i,i,1]=0.1
    if i%10>0:
        P[i,i-1,1]=0.1
    else:
        P[i,i,1]=0.1

    # Left is -1
    if (i>9):
        P[i,i-10,2]=0.1
    else:
        P[i,i,2]=0.1

    if i%10>0:
        P[i,i-1,2]=0.8
    else:
        P[i,i,2]=0.8

    if (i<90):
        P[i,i+10,2]=0.1
    else:
        P[i,i,2]=0.1

    # Right is +1
    if (i>9):
        P[i,i-10,3]=0.1
    else:
        P[i,i,3]=0.1

    if i%10<9:
        P[i,i+1,3]=0.8
    else:
        P[i,i,3]=0.8

    if (i<90):
        P[i,i+10,3]=0.1
    else:
        P[i,i,3]=0.1   

# Wormholes
for i in [32,42,52,62]:
    for j in range(4):
        P[i,i+1,j]=0
        P[i,i-1,j]=0
        P[i,i+10,j]=0
        P[i,i-10,j]=0
        P[i,0,j]=1

i=17
for j in range(4):
    P[i,i+1,j]=0
    P[i,i-1,j]=0
    P[i,i+10,j]=0
    P[i,i-10,j]=0
    P[i,97,j]=1

# Terminal stage

P[terminal,terminal,:]=1
if (terminal%10)<9:
    P[terminal,terminal+1,:]=0
if (terminal%10)>0:
    P[terminal,terminal-1,:]=0
if (terminal<90):
    P[terminal,terminal+10,:]=0
if (terminal>9):
    P[terminal,terminal-10,:]=0

bel.P=P

r=-1*np.ones((100,100,4))
r[:,terminal,:]=100
# Collect reward only once
r[terminal,terminal,:]=0

bel.r=r

def quiver_actions(actions,terminal=args.terminal,stage=0, save_path=None,supress=False):
    '''
    Plot a quiver plot of the policy
    '''
    def _action_u(u):
        '''
        Horz quiver
        -1,1 if u == left or right
        0 else
        '''
        if u==2:
            return -1
        elif u==3:
            return 1
        else:
            return 0
    def _action_v(u):
        '''
        Vert quiver
        -1,1 if u == down or up
        0 else
        '''
        if u==0:
            return 1
        elif u==1:
            return -1
        else:
            return 0
    
    X=Y=np.arange(0.5,10.5,1)
    U= np.array([_action_u(a) for a in actions]).reshape((10,10))
    V= np.array([_action_v(a) for a in actions]).reshape((10,10))
    q=plt.quiver(X,Y,U,V)
    plt.quiverkey(q,X=8, Y=8, U=1,label='Quiver key, length = 1', labelpos='E')
    plt.title(f"Quiver state plot at stage {stage}")

    major_ticks = np.arange(0, 10, 1)

    # Wormholes 1
    for j in range(3,8):
        plt.scatter(2.5,j+0.5,s=225,color=colors['red'])
        plt.text(2.5, j + 0.5, 'IN1')
    # Exit 1
    plt.scatter(0.5,0.5,s=225,color=colors['maroon'])
    plt.text(0.5, 0.5, 'OUT1')

    # Wormholes 2
    plt.scatter(7.5,1.5,s=225,color=colors['grey'])
    plt.text(7.5, 1.5, 'IN2')
    plt.scatter(7.5,9.5,s=225,color=colors['lightgrey'])
    plt.text(7.5, 9.5, 'OUT2')

    # Terminal State
    a=args.terminal//10+0.5
    b=args.terminal%10+0.5
    plt.scatter(b,a,s=256,color=colors['green'])
    plt.text(b,a, 'TERMINAL')

    plt.xlim((0,10))
    plt.ylim((0,10))
    plt.xticks(major_ticks)
    plt.yticks(major_ticks)

    plt.grid(True)

    if save_path:
        plt.savefig(os.path.join(save_path,f"quiver-{stage}.png"))
    if not supress:
        plt.show()
    else:
        plt.close()

def plot_heatmaps(J, terminal= args.terminal, stage=0,save_path=None,supress=False):
    '''
    Plot a heatmap for a single J
    '''
    # Plot a heatmap
    im=plt.imshow(J.reshape((10,10))[::-1,:],cmap="jet")

    CB = plt.colorbar(im, shrink=0.8, extend='both')


    # Set ticks
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))

    # Loop over data dimensions and create text annotations.
    for i in range(9,-1,-1):
        for j in range(10):
            text = plt.text(j, i, f"{J.reshape((10,10))[::-1,:][i, j]:.2f}",\
            ha="center", va="center", color="w", fontsize=6)

    plt.title(f"J value at stage {stage}")

    # Save plot
    if save_path:
        plt.savefig(os.path.join(save_path,f"J-heatmap-{stage}.png"))

    if not supress:
        plt.show()
    else:
        plt.close()

def plot_convergence(J_array, terminal= args.terminal, stage=0, save_path=None,supress=False):
    '''
    Plot rewards, at stages. 
    Stages are inverted here for convinience, but nonetheless holds.
    '''
    J_array=np.array(J_array)
    J_diff=np.max(np.abs(J_array[1:]-J_array[:-1]),axis=1)
    iters=np.arange(1,len(J_diff)+1)
    plt.plot(iters,J_diff)
 
    plt.grid()

    for j in range(len(J_diff)):
        if (j<10 and j%3==0) or (j%10==0):
            plt.text(j+1.15,J_diff[j]+0.15,s=f'value {J_diff[j]:.2f}')

    plt.title("$max_s |J_{i+1}(s) − J_i(s)|$ vs iterations.")

    # Save plot
    if save_path:
        plt.savefig(os.path.join(save_path,f"convergence-till-{stage}.png"))

    if not supress:
        plt.show()
    else:
        plt.close()
    
J_array,actions_array=bel.optimal_policy(verbose=True,epsilon=1e-9)

# Save plots
N=len(actions_array)

path=f"logs/t={args.terminal}_N={args.stages}"

if not os.path.exists(path):
    os.mkdir(path)

for i in range(N):
    # Quiver plots
    quiver_actions(actions_array[N-i-1],stage=N-i-1,\
    save_path=path,\
    supress=bool(args.supress))

    # J plots
    plot_heatmaps(J_array[N-i-1],stage=N-i-1,\
    save_path=path,\
    supress=bool(args.supress))

plot_convergence(J_array,stage=N,\
    save_path=path,\
    supress=bool(args.supress))