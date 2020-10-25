import sys
import gym
import argparse
import numpy as np
import atari_py
from ddqn_game_model import DDQNTrainer
from atari_wrappers import MainGymWrapper

NO_FRAMES= 4
FRAME_SIZE = 84
INPUT_SHAPE = (NO_FRAMES, FRAME_SIZE, FRAME_SIZE)

class Atari:

    def __init__(self):
         game_mode, render, total_step_limit, total_run_limit, clip = self.parseArgs()
         env_name = "SpaceInvadersDeterministic-v4"  
         env = MainGymWrapper.wrap(gym.make(env_name))
         self.run(self.game_model(game_mode, env.action_space.n), env, render, total_step_limit, total_run_limit, clip)



    def run(self, game_model, env, render, total_step_limit, total_run_limit, clip):
        

        run = 0
        total_step = 0
        while True:
            if(total_run_limit) !=None and run >= total_run_limit:
                print ("Reached total run limit of: " + str(total_run_limit))
                sys.exit(0)

            run += 1
            current_state = env.reset()
            step = 0
            score = 0
            while True:
                if total_step >= total_step_limit:
                    print("Reached total step limit of: " + str(total_step_limit))
                    sys.exit(0)
                total_step += 1
                step += 1


                env.render()

                action = game_model.move(current_state)
                next_state, reward, terminal, info = env.step(action)
                if clip:
                    np.sign(reward)
                score += reward
                game_model.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state          # chang the current state of the game to the state the game is after performing the action

                game_model.step_update(total_step)

                if terminal:
                    game_model.save_run(score, step, run)
                    break
    
 
  


    def parseArgs(self):
        parser = argparse.ArgumentParser()
        
        parser.add_argument("-m", "--mode", help="Choose the modes: ddqn_train, ddqn_test. Default is 'ddqn_training'.", default="ddqn_training")
        parser.add_argument("-tsl", "--total_step_limit", help="Choose how many total steps (frames visible by agent) should be performed. Default is '5000000'.", default=5000000, type=int)
        parser.add_argument("-trl", "--total_run_limit", help="Choose after how many runs we should stop. Default is None (no limit).", default=None, type=int)
    
        args = parser.parse_args()
        game_mode = args.mode
        render= True  # Render the game
        
        
        total_step_limit = args.total_step_limit
        total_run_limit = args.total_run_limit
        clip = True
       
        print("Selected mode: " + str(game_mode))
        
        print("Total step limit: " + str(total_step_limit))
        print("Total run limit: " + str(total_run_limit))
        return game_mode, render, total_step_limit, total_run_limit, clip

    def game_model(self, game_mode, action_space):
        if game_mode == "ddqn_training":
            return DDQNTrainer(INPUT_SHAPE, action_space)
        elif game_mode == "ddqn_testing":
            return DDQNSolver( INPUT_SHAPE, action_space)
       
        else:
            print("Unrecognized mode. Use --help")
            exit(1)


if __name__ == "__main__":
    Atari()