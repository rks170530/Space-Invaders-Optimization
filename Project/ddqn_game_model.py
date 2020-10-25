import numpy as np
import os
import random
import shutil
from statistics import mean
from base_game_model import BaseGameModel
from CNN import getWeights,initialiseNetwork,predict,predict_target

# All of the Hyper Parameters.
VALUE_FRAME_SIZE = 900000
VALUE_BATCH_SIZE = 32
VALUE_GAMMA = 0.99

VALUE_REPLAY_SIZE_START = 50000
VALUE_TARGET_NW_UPDATE_FREQUENCY = 40000
VALUE_MODEL_PERSISTENCE_UPDATE_FREQUENCY = 10000
VALUE_TRAINING_FREQUENCY = 4


VALUE_MIN_EXPLORATION = 0.1
VALUE_MAX_EXPLORATION = 1.0
VALUE_STEPS_EXPLORATION = 850000
EXPLORATION_DECAY = (VALUE_MAX_EXPLORATION-VALUE_MIN_EXPLORATION)/VALUE_STEPS_EXPLORATION

VALUE_TEST_EXPLORATION = 0.02






# Did NOT change Line 30 to Line 85, part of CNN.
class DDQNGameModel(BaseGameModel):

    def __init__(self, mode_name, input_shape, action_space, logger_path, model_path):
        BaseGameModel.__init__(self,
                               mode_name,
                               logger_path,
                               input_shape,
                               action_space)




# =============================================================================
# class DDQNSolver(DDQNGameModel):
# 
#     def __init__(self, game_name, input_shape, action_space):
#         #testing_model_path = "./output/neural_nets/" + game_name + "/ddqn/testing/model.h5"
#         assert os.path.exists(os.path.dirname(testing_model_path)), "No testing model in: " + str(testing_model_path)
#         DDQNGameModel.__init__(self,
#                                game_name,
#                                "DDQN testing",
#                                input_shape,
#                                action_space,
#                                "./output/logs/" + game_name + "/ddqn/testing/" + self._get_date() + "/",)
# 
#     def move(self, state):
#         if np.random.rand() < VALUE_TEST_EXPLORATION:
#             return random.randrange(self.action_space)
#         q_values = self.ddqn.predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0), batch_size=1)
#         return np.argmax(q_values[0])
# =============================================================================


class DDQNTrainer(DDQNGameModel):

    def __init__(self, input_shape, action_space):
        DDQNGameModel.__init__(self,
                               "DDQN training",
                               input_shape,
                               action_space,
                               "./output/logs/" + "/ddqn/training/" + self._get_date() + "/",
                               "./output/neural_nets/"  + "/ddqn/" + self._get_date() + "/model.h5")

#        if os.path.exists(os.path.dirname(self.model_path)):
#            shutil.rmtree(os.path.dirname(self.model_path), ignore_errors=True)
#        os.makedirs(os.path.dirname(self.model_path))

        #self.ddqn_target = C(self.input_shape, action_space).model
        self.parameters=self._reset_target_network()
        #print('Weights 1 ',self.parameters)






        self.epsilonValue = VALUE_MAX_EXPLORATION
        self.memoryList = []

    def move(self, state):

        if len(self.memoryList) < VALUE_REPLAY_SIZE_START:
            return random.randrange(self.action_space)              # While training the network for first time, return a random action to perform

        if np.random.rand() < self.epsilonValue:
            return random.randrange(self.action_space)     # if random number < eplison, choose a action randomly

        # Need to implement the Predict function in your CNN based on the Karas library.
        qValues =predict(np.expand_dims(np.asarray(state).astype(np.float64), axis=0))     #Done
        
        #print('qq value ',qValues)

        maxQValue = np.argmax(qValues[0])
        return maxQValue

    # Similar functionality to the Replay buffer, just a different implementation. Choose whichever you prefer.
    def remember(self, currentState, stateAction, rewardValue, nextState, checkEnd):
        self.memoryList.append({"current_state": currentState,
                            "action": stateAction,
                            "reward": rewardValue,
                            "next_state": nextState,
                            "terminal": checkEnd})

        if len(self.memoryList) > VALUE_FRAME_SIZE:
            self.memoryList.pop(0)

    # Updates the Q values, using the inputTotalSteps to be the total steps defined.
    def step_update(self, inputTotalSteps):
        count=0

        

        if len(self.memoryList) < VALUE_REPLAY_SIZE_START:
            return

        if inputTotalSteps % VALUE_TRAINING_FREQUENCY == 0:
            count+=1

            inputList = self._train()

            lossValue = inputList[0]
            #accuracyValue = inputList[1]
            avgMaxQValue = inputList[1]
            
            print()
            print('Error after training for ',count,' times: ',lossValue )

            self.logger.add_q(avgMaxQValue)
            #self.logger.add_accuracy(accuracyValue)
            self.logger.add_loss(lossValue)

        self._update_epsilon()

        if inputTotalSteps % VALUE_MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
            self._save_model()

        if inputTotalSteps % VALUE_TARGET_NW_UPDATE_FREQUENCY == 0:
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilonValue))
            print('{{"metric": "total_step", "value": {}}}'.format(inputTotalSteps))
            self.parameters=self._reset_target_network()

    def _train(self):
        sampleBatch = np.asarray(random.sample(self.memoryList, VALUE_BATCH_SIZE))

        if len(sampleBatch) < VALUE_BATCH_SIZE:
            return

        qValueList = []
        maxQValueList = []
        currentStatesList = []

        for batchEntry in sampleBatch:

            # Trains against itself by getting the current state and the current state list.
            currentState = np.expand_dims(np.asarray(batchEntry["current_state"]).astype(np.float64), axis=0)
            currentStatesList.append(currentState)

            # Maximizes the reward values of the next state list.
            nextState = np.expand_dims(np.asarray(batchEntry["next_state"]).astype(np.float64), axis=0)
            nextStatePrediction = predict_target(nextState,self.parameters).ravel()
            nextQValue = np.max(nextStatePrediction)

            
            qList = list((predict(currentState))[0])

            # Similar to replay_buffer, checks if the state is at the end condition.
            if batchEntry["terminal"] == False:
                qList[batchEntry["action"]] = batchEntry["reward"] + (VALUE_GAMMA * nextQValue)

            if batchEntry["terminal"] == True:
                qList[batchEntry["action"]] = batchEntry["reward"]

            else:
                print("Error, batchEntry is incorrect.")

            maxQValueList.append(np.max(qList))
            qValueList.append(qList)

        # Need to implement the Fit function in your CNN based on the Karas library.
        fitDDQN = initialiseNetwork(np.asarray(currentStatesList).squeeze(),
                            np.asarray(qValueList).squeeze() )


        avgMaxQValue = mean(maxQValueList)
        error = fitDDQN
        #accuracyValue = fitDDQN.history["acc"][0]

        returnList = [error, avgMaxQValue]
        return returnList

    # Updates the Epsilon value.
    def _update_epsilon(self):
        self.epsilonValue = self.epsilonValue - EXPLORATION_DECAY
        maxEpsilonValue = max(VALUE_TEST_EXPLORATION, self.epsilonValue)
        self.epsilonValue = maxEpsilonValue

    # Resetting the target NW, I believe it's similar to the clear() function in the replay_buffer.py,
    # but using the Keras library. Please check/implement using the Keras library.
    def _reset_target_network(self):
        return getWeights()
    