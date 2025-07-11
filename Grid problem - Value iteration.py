#Plan: 
# 1. go through every action at every state
# 2. Apply the Bellman's equation at every state - that should generate me the best an array (one value per state)
#   (Reevaluating all the options at each state, but only retain the highest value, i.e. best action)
# 3. Stop once it stabilise/converge - we can measure this by monitoring the change in V values

import numpy as np

# 1. set up the model - Value iteration is a model-based RL method
#       I will use the same environment as the one defined in the Q-learning one


# table of possible actions from each state:

choice = np.array([[0,-1], [0,+1], [-1,0],[+1,0]])

action = np.zeros((25,4))

for x in range (0,5):
    for y in range(0,5):

        s = x + 5*y
        for a in range (0,4):
        
        #0=up(0,-1)
        #1=down(0,+1)
        #2=left(-1,0)
        #3=right(+1,0)

            next_x = x + choice[a,0]
            next_y = y + choice[a,1]
            action[s,a] = next_x + 5*next_y

            if next_x < 0 or next_x >4 or next_y < 0 or next_y >4:
                action[s,a] = s

#copying over the reward system
def reward_system (current_state,next_state): 

    # add a distance based reward - to help it to spot whether it is getting closer
    # convert into coordinate to enable me to calculate the distance between current position
    # and the destination
    current_y = current_state // 5 
    current_x = current_state % 5
    current_dist = ((current_y - 5)**2 + (current_x -5)**2)**1/2
    next_y = next_state // 5
    next_x = next_state % 5
    next_dist = ((next_y - 5)**2 + (next_x -5)**2)**(1/2)
    print(current_dist-next_dist)

    if next_state == 7 or next_state == 8 or next_state == 10 or next_state == 13 or next_state == 15:
        reward = int(-50)
    elif next_dist < current_dist :
        reward = 10
    if next_state == 24:
        reward = 1000
    else: 
        reward = int(-20)

    return reward


V_table = np.zeros((25))
change = np.full(25,float(1))
V_table[24] = 1000
avg_change = 1
while avg_change > 0.05:
    for s in range (0,24): 
        max_value = -100
        #Q_temp = a temperary array to store the Q value of all 4 action
        #for the system to then compare and determine the V value
        # V value will the weighted average of all the decison based on probability of action taken
        # we will fixed the final destination to a value of 100 to ensure it moves towards it
        Q_temp = np.zeros(4)
        for a in range(0,4): 

            # 1. find the V value for all four action

            Q_temp[a] = reward_system(s,action[s,a]) + 0.9*V_table[int(action[s,a])]
            
        
        # 2. sorting the q_values based on asscending order

        Q_temp = sorted(Q_temp)

        V_new = 0.925*Q_temp[0] + 0.025*Q_temp[1] + 0.025*Q_temp[2] + 0.025*Q_temp[3]
            

        # finding the change in V values - later useful in determining convergence
        if V_new != V_table[s]:
            # this helps the system to avoid DNE when V values = 0
            if V_table[s] != 0: 
                change[s] = abs(max_value - V_table[s])/V_table[s]
            if V_table[s] == 0: 
                change[s] = 1
            #updating the V value
            V_table[s] = V_new

    #parameter to determine convergence
    avg_change = sum(change)/25

# I am now determining the best policy using the value table

# starting 0,0 

position = 0 
policy = []
policy.append(position)

#compare the all the next possible state at the current state
# determine which state has the highest V value 
#which will be the one the system will move onto

print(V_table)
while position != 24:
    comparison_table = np.zeros(4) 
    for i in range (0,4):
        next_state = int(action[int(position),int(i)])
        comparison_table[i] = int(V_table[next_state])
    
    print(comparison_table)

    best_next_state_index = np.argmax(comparison_table)
    print("index:",best_next_state_index)

    best_next_state = action[int(position),best_next_state_index]
    print(best_next_state)

    policy.append(int(best_next_state))

    position = best_next_state

V_grid = V_table.reshape((5, 5))
print(V_grid)

print(policy)
