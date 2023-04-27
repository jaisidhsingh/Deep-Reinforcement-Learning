import common.game_constants as game_constants
import common.game_state as game_state
import pygame
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm 
import math
import random
from copy import deepcopy


class KeyboardController:
    def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
        keys = pygame.key.get_pressed()
        action = game_state.GameActions.No_action
        if keys[pygame.K_LEFT]:
            action = game_state.GameActions.Left
        if keys[pygame.K_RIGHT]:
            action = game_state.GameActions.Right
        if keys[pygame.K_UP]:
            action = game_state.GameActions.Up
        if keys[pygame.K_DOWN]:
            action = game_state.GameActions.Down
    
        return action


class AIController:
### ------- You can make changes to this file from below this line --------------
    def __init__(self) -> None:

        self.state_dims = 12 + game_constants.ENEMY_COUNT * 6
        self.action_dims = 5
        self.learning_rate = 1e-9

        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.95

        self.q_network = nn.Sequential(*[
            nn.Linear(self.state_dims, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, self.action_dims),
        ])
        self.optimizer = torch.optim.SGD(
            self.q_network.parameters(), 
            lr=self.learning_rate,
            momentum=0.0
        )
        # NO EXPERIENCE REPLAY BUFFER STORING TRANSITIONS

    def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
        # This function should select the best action at a given state
    
        # make state tensor first
        player_features = [
            state.PlayerEntity.entity.x,
            state.PlayerEntity.entity.y,
            state.PlayerEntity.entity.height,
            state.PlayerEntity.entity.width,
            state.PlayerEntity.velocity.x,
            state.PlayerEntity.velocity.y,
            state.PlayerEntity.friction,
            state.PlayerEntity.acc_factor
        ]

        goal_features = [
            state.GoalLocation.x,
            state.GoalLocation.y,
            state.GoalLocation.height,
            state.GoalLocation.width,
        ]

        enemy_features = [
            [
                e.entity.x, e.entity.y, 
                e.entity.height, e.entity.width,
                e.velocity.x, e.velocity.y
            ] for e in state.EnemyCollection
        ]

        state_list = []
        for feature_list in [player_features, goal_features]:
            for f in feature_list:
                state_list.append(float(f))
        
        for e in enemy_features:
            for f in e:
                state_list.append(float(f))

        state_array = np.array(state_list) 
        state_tensor = torch.from_numpy(state_array)
        state_tensor = state_tensor.unsqueeze(0).float()
         # add batch for forward pass through torch nn
        
        action_predictions = self.q_network(state_tensor)
        action_idx = torch.argmax(action_predictions, 1)

        action2take = torch.argmax(torch.softmax(action_predictions, dim=1), 1).item()
        return game_state.GameActions(int(action2take))

    def forwardPass(self, state:game_state.GameState):
        # make state tensor first
        player_features = [
            state.PlayerEntity.entity.x,
            state.PlayerEntity.entity.y,
            state.PlayerEntity.entity.height,
            state.PlayerEntity.entity.width,
            state.PlayerEntity.velocity.x,
            state.PlayerEntity.velocity.y,
            state.PlayerEntity.friction,
            state.PlayerEntity.acc_factor
        ]

        goal_features = [
            state.GoalLocation.x,
            state.GoalLocation.y,
            state.GoalLocation.height,
            state.GoalLocation.width,
        ]

        enemy_features = [
            [
                e.entity.x, e.entity.y, 
                e.entity.height, e.entity.width,
                e.velocity.x, e.velocity.y
            ] for e in state.EnemyCollection
        ]

        state_list = []
        for feature_list in [player_features, goal_features]:
            for f in feature_list:
                state_list.append(float(f))
        
        for e in enemy_features:
            for f in e:
                state_list.append(float(f))

        state_array = np.array(state_list) 
        state_tensor = torch.from_numpy(state_array)
        state_tensor = state_tensor.unsqueeze(0).float()
        # add batch for forward pass through torch nn
        return self.q_network(state_tensor)

    def GetDistance(self, player_entity, goal):
        xsq = ((player_entity.x+ 0.5*player_entity.width) - (goal.x+ 0.5*goal.width))**2
        ysq = ((player_entity.y+ 0.5*player_entity.height) - (goal.y+0.5*goal.height))**2
        return math.sqrt(xsq + ysq)

    def computeQTargets(self, next_state, reward, done):
        q_values = self.forwardPass(next_state).detach().max(1)[0].unsqueeze(0)
        return reward + 0.95*q_values# *(1 - done) #+ done * enemy_term
    
    def computeQExpected(self, state, action):
        q_exp = self.forwardPass(state)[:, action] #.gather(1, actions.long())
        return q_exp.view((1, 1))

    def computeLoss(self, state, action, next_state, reward, done):
        q_targets = self.computeQTargets(next_state, reward, done)
        q_expected = self.computeQExpected(state, action)

        loss = torch.nn.functional.mse_loss(q_expected, self.learning_rate*q_targets)
        return loss

    def epsilonGreedyStrategy(self, current_state, steps_done):
        random_sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start-self.epsilon_end)
        eps_threshold = eps_threshold*math.exp(-1*steps_done / self.epsilon_decay)

        eps_threshold = 0.5

        if random_sample > eps_threshold:
            action = self.GetAction(current_state) # Select best action
        else:
            random_choice = random.choice([0, 1, 2, 3, 4])
            action = game_state.GameActions(int(random_choice))

        return action

    def TrainModel(self):
        steps_done = 0
        episodes = 1e3

        state = game_state.GameState()
        current_state = deepcopy(state)
        
        while steps_done <= episodes:
            # not finished (init)
            done = 0
            # action = explore_vs_exploit()
            action = self.epsilonGreedyStrategy(current_state, steps_done)
            steps_done += 1
            # update to the next state
            obs = state.Update(action) # obtain the observation made due to your action

            # start the loss
            loss = 0

            # formulate the reward
            enemy_term = 0 
            if obs.value == -1:
                done = 1
                enemy_term = -85
            # straight line distance based heuristic
            e = 1e-2
            dpg = self.GetDistance(state.PlayerEntity.entity, state.GoalLocation)
            goal_scale_factor = 8e3
            heuristic_term = goal_scale_factor*(1/dpg+e)

            idle_scale_factor = 1e-2
            offset = -1*idle_scale_factor*dpg -2 + 4e-3*dpg
            # offset = -5

            reward_at_this_step = offset + heuristic_term + enemy_term

            # edit reward if we die via an enemy

            loss = self.computeLoss(
                current_state, 
                action.value, 
                state, 
                reward_at_this_step, 
                done 
            )

            loss.backward()
            self.optimizer.step()
            current_state = deepcopy(state)


### ------- You can make changes to this file from above this line --------------

    # This is a custom Evaluation function. You should not change this function
    # You can add other methods, or other functions to perform evaluation for
    # yourself. However, this evalution function will be used to evaluate your model
    def EvaluateModel(self):
        attacked = 0
        reached_goal = 0
        state = game_state.GameState()
        for _ in tqdm(range(100000)):
            action = self.GetAction(state)
            obs = state.Update(action)
            if(obs==game_state.GameObservation.Enemy_Attacked):
                attacked += 1
            elif(obs==game_state.GameObservation.Reached_Goal):
                reached_goal += 1
        return (attacked, reached_goal)