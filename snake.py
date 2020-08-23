import gym
from simple_dqn_torch_2020_snake import Agent
import snake_gym
import numpy as np
import torch as T
import sys
import time

if __name__ == '__main__':
    env = gym.make('Snake-v0')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01,
                  input_dims=[51], lr=0.001)
    if(int(sys.argv[1])==1):    
        print("Loading "+sys.argv[2]+" model")
        agent.Q_eval = T.load('./Trained_model/'+sys.argv[2]+'_eval.pt')
        agent.Q_next = T.load('./Trained_model/'+sys.argv[2]+'_next.pt')
        file = open('./Trained_model/eps.txt','r')
        agent.epsilon = float(file.read())
        file.close()
    scores, eps_history = [], []
    n_games = int(sys.argv[2])
    
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        score = 0
        pellets = 0
        survived = 0
        while not done:
            survived+=1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if(reward==100):
                pellets+=1
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
                                    
            if(int(sys.argv[1])==2):
                agent.learn()
            else:
                env.render()
                if(pellets>0):
                    time.sleep(0.05)
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        print('Score:',score,"\t\t Pellets: ",pellets,"\t\t Steps: ",survived)
        if(int(sys.argv[1])==2):
            if(i%100==0):
                T.save(agent.Q_eval,'./Trained_model/'+str(i)+'_eval.pt')
                T.save(agent.Q_next,'./Trained_model/'+str(i)+'_next.pt')
            file = open('./Trained_model/eps.txt','w')
            file.write(str(agent.epsilon))
            file.close()
