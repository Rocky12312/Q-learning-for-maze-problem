from maze_env import Maze
from RL_agent import QLearningTable
import matplotlib.pyplot as plt

#The number of episodes are the number of time the agent does some actions before getting to goal or say the hell points
episode_count = 50 
episodes = range(episode_count)
#Number of movements happened in each episode
movements = []
#The gained reward in each episode
rewards = []

#Creating a function to run the experiments
def run_experiment():

    for episode in episodes:
        print("Episode %s/%s." %(episode+1, episode_count))
        #Initial observation;
        observation = env.reset()
        #Initially setting the number of moves to zero
        moves = 0

        while True:
            #Fresh environment
            env.render()

            #Q-learning chooses action based on observation
            #We convert observation to str since we want to use them as index for our DataFrame.
            action = q_learning_agent.choose_action(str(observation)) # ToDo: call choose_action() method from the agent QLearningTable instance

            #RL takes action and gets next observation and reward
            observation_, reward, done = env.get_state_reward(action) # ToDo: call get_state_reward() method from Maze environment instance
            moves +=1

            #RL learn from the above transition,
            #Update the Q value for the given tuple
            q_learning_agent.learn(str(observation), action, reward, str(observation_))# ToDo: call learn method from Q-learning agent instance, passing (s, a, r, s') tuple

            #Consider the next observation
            observation = observation_

            #Break while loop when end of the episode accur
            if done:
                movements.append(moves)
                rewards.append(reward)
                print("Reward: {0}, Moves: {1}".format(reward, moves))
                break

    print('game over!')
    #Showing the results
    plot_reward_movements()

def plot_reward_movements():
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(episodes, movements)
    plt.xlabel("Episode")
    plt.ylabel("# Movements")

    plt.subplot(2,1,2)
    plt.step(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("rewards_movements_q_learn.png")
    plt.show()


if __name__ == "__main__":

    #Creating the maze environment
    env = Maze()

    #Creating the agent
    q_learning_agent = QLearningTable(actions=list(range(env.n_actions))) 

    #Call run_experiment() function once after given time
    env.window.after(10, run_experiment)

    #The infinite loop used to run the application, wait for an event to occur and process the event till the window is not closed.
    env.window.mainloop()
