# reset all the variables from previous cells
%reset -f

from abc import ABC, abstractmethod
from dice_game import DiceGame
import numpy as np


class DiceGameAgent(ABC):
    def __init__(self, game):
        self.game = game

    @abstractmethod
    def play(self, state):
        pass


class AlwaysHoldAgent(DiceGameAgent):
    def play(self, state):
        return (0, 1, 2)


class PerfectionistAgent(DiceGameAgent):
    def play(self, state):
        if state == (1, 1, 1) or state == (1, 1, 6):
            return (0, 1, 2)
        else:
            return ()


def play_game_with_agent(agent, game, verbose=False):
    state = game.reset()

    print(state)
    if(verbose): print(f"Testing agent: \n\t{type(agent).__name__}")
    if(verbose): print(f"Starting dice: \n\t{state}\n")

    game_over = False
    actions = 0
    while not game_over:
        action = agent.play(state)
        actions += 1

        if(verbose): print(f"Action {actions}: \t{action}")
        _, state, game_over = game.roll(action)
        if(verbose and not game_over): print(f"Dice: \t\t{state}")

    if(verbose): print(f"\nFinal dice: {state}, score: {game.score}")

    return game.score


def main():
    # random seed makes the results deterministic
    # change the number to see different results
    # or delete the line to make it change each time it is run
    np.random.seed(1)

    game = DiceGame()

    agent1 = AlwaysHoldAgent(game)
    play_game_with_agent(agent1, game, verbose=True)

    print("\n")

    agent2 = PerfectionistAgent(game)
    play_game_with_agent(agent2, game, verbose=True)


if __name__ == "__main__":
    main()





class MyAgent(DiceGameAgent):

    def __init__(self, game):


        actions=game.actions #list of all possible actions
        V={} #create a dictionary for value functions
        self.Q={} #create a dictionary for policy
        gamma=1 #initialise gamma
        theta=0.1 #inisitilist theta
        V0=0
        epsilon=0

        for statess in game.states:
            V.update({statess:V0}) #create dictionary for V values of 0
            self.Q.update({statess:0}) #create dictionary for Q values

        if epsilon<10:

            V_copy=V.copy()
            for statess in game.states: #iterate through every possible game state
                best_action_value=0

                for action in actions: #iterate through all actions

                    expected_reward=0
                    expected_value=0
                    action_value=0

                    #find the next states for current action and state
                    states, game_over, reward, probabilities = game.get_next_states((action), (statess))
                    for state, probability in zip(states, probabilities):

                        if state==None:
                            game_score=game.final_scores.get(statess)
                        else:
                            game_score=game.final_scores.get(state)

                        #calculate expected reward
                        expected_reward+=(game_score)*probability

                        #find the value of V in the V dictionary
                        if state==None:
                            lookupV=V.get((statess))
                        else:
                            lookupV=V.get((state))

                        #calculate expected value
                        expected_value+=lookupV*probability

                        #calculate the action value, V
                    action_value=expected_reward + gamma*expected_value

                    #find the action that gives the larger value and set that to best_action
                    if action_value>best_action_value:
                        best_action_value=action_value
                        best_action=action

                #add the state with the best value and action to the dictionaries
                V_copy.update({statess:best_action_value})
                self.Q.update({statess:best_action})

                #if the difference in V values converges break the loop
                change=V_copy.get(statess)-V.get(statess)
                if change<theta:
                    break

            V=V_copy
            epsilon+=1

                                    
    def play(self,state):
        action=self.Q.get((state))

        return action

