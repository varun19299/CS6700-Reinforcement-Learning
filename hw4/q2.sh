# Runs all the necessary python args

# In suppressed mode
cd ~/Documents/SEM-5/RL/assignments/hw2

# Terminal as 99
python3 q2.py --terminal 99 --stages 10 --supress 1
python3 q2.py --terminal 99 --stages 25 --supress 1
python3 q2.py --terminal 99 --stages -1 --supress 1

# Terminal as 3
python3 q2.py --terminal 3 --stages 10 --supress 1
python3 q2.py --terminal 3 --stages 25 --supress 1
python3 q2.py --terminal 3 --stages -1 --supress 1

# logs at ./logs