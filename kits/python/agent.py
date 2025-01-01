from lux.utils import direction_to
import sys
import numpy as np
import logging

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()
        self.non_zero_tile_positions = {}

        if self.player == "player_0":
            logging.basicConfig(filename='player_0.log', level=logging.INFO, filemode='w')

    def get_unit_mask(self, obs):
        return np.array(obs["units_mask"][self.team_id])

    def get_unit_positions(self, obs):
        return np.array(obs["units"]["position"][self.team_id])

    def get_unit_energys(self, obs):
        return np.array(obs["units"]["energy"][self.team_id])

    def get_observed_relic_node_positions(self, obs):
        return np.array(obs["relic_nodes"])

    def get_observed_relic_nodes_mask(self, obs):
        return np.array(obs["relic_nodes_mask"])

    def get_team_points(self, obs):
        return np.array(obs["team_points"])

    def get_sensor_mask(self, obs):
        return np.array(obs["sensor_mask"])

    def get_map_features_energy(self, obs):
        return np.array(obs["map_features"]["energy"])

    def get_map_features_tile_type(self, obs):
        return np.array(obs["map_features"]["tile_type"])

    def get_team_wins(self, obs):
        return np.array(obs["team_wins"])

    def get_steps(self, obs):
        return obs["steps"]

    def get_match_steps(self, obs):
        return obs["match_steps"]

    def get_max_units(self):
        return self.env_cfg["max_units"]

    def get_match_count_per_episode(self):
        return self.env_cfg["match_count_per_episode"]

    def get_max_steps_in_match(self):
        return self.env_cfg["max_steps_in_match"]

    def get_map_height(self):
        return self.env_cfg["map_height"]

    def get_map_width(self):
        return self.env_cfg["map_width"]

    def get_num_teams(self):
        return self.env_cfg["num_teams"]

    def get_unit_move_cost(self):
        return self.env_cfg["unit_move_cost"]

    def get_unit_sap_cost(self):
        return self.env_cfg["unit_sap_cost"]

    def get_unit_sap_range(self):
        return self.env_cfg["unit_sap_range"]

    def get_unit_sensor_range(self):
        return self.env_cfg["unit_sensor_range"]

    def log_state(self, step, obs):
        if self.player == "player_0":
            logging.info(f"Step: {step}")
            
            # Log Unit Info
            logging.info("Logging Unit Info:")
            logging.info(f"Unit Mask: {self.get_unit_mask(obs)}")
            logging.info(f"Unit Positions: {self.get_unit_positions(obs)}")
            logging.info(f"Unit Energys: {self.get_unit_energys(obs)}")
            
            # Log Map Info
            logging.info("Logging Map Info:")
            logging.info(f"Observed Relic Node Positions: {self.get_observed_relic_node_positions(obs)}")
            logging.info(f"Observed Relic Nodes Mask: {self.get_observed_relic_nodes_mask(obs)}")
            logging.info(f"Sensor Mask: {self.get_sensor_mask(obs)}")
            logging.info(f"Map Features Energy: {self.get_map_features_energy(obs)}")
            logging.info(f"Map Features Tile Type: {self.get_map_features_tile_type(obs)}")
            
            # Log Game Info
            logging.info("Logging Game Info:")
            logging.info(f"Team Points: {self.get_team_points(obs)}")
            logging.info(f"Team Wins: {self.get_team_wins(obs)}")
            logging.info(f"Steps: {self.get_steps(obs)}")
            logging.info(f"Match Steps: {self.get_match_steps(obs)}")
            
            # Log Environment Configuration
            logging.info("Logging Environment Configuration:")
            logging.info(f"Max Units: {self.get_max_units()}")
            logging.info(f"Match Count Per Episode: {self.get_match_count_per_episode()}")
            logging.info(f"Max Steps In Match: {self.get_max_steps_in_match()}")
            logging.info(f"Map Height: {self.get_map_height()}")
            logging.info(f"Map Width: {self.get_map_width()}")
            logging.info(f"Num Teams: {self.get_num_teams()}")
            logging.info(f"Unit Move Cost: {self.get_unit_move_cost()}")
            logging.info(f"Unit Sap Cost: {self.get_unit_sap_cost()}")
            logging.info(f"Unit Sap Range: {self.get_unit_sap_range()}")
            logging.info(f"Unit Sensor Range: {self.get_unit_sensor_range()}")

            # Record non-zero tile positions
            self.record_non_zero_tile_positions(step, self.get_map_features_tile_type(obs))

    def record_non_zero_tile_positions(self, step, tile_type):
        non_zero_positions = np.argwhere(tile_type > 0)
        non_zero_values = tile_type[non_zero_positions[:, 0], non_zero_positions[:, 1]]
        self.non_zero_tile_positions[step] = list(zip(non_zero_positions.tolist(), non_zero_values.tolist()))
        logging.info(f"Non-zero Tile Positions at Step {step}: {self.non_zero_tile_positions[step]}")

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        unit_mask = self.get_unit_mask(obs)
        unit_positions = self.get_unit_positions(obs)
        unit_energys = self.get_unit_energys(obs)
        observed_relic_node_positions = self.get_observed_relic_node_positions(obs)
        observed_relic_nodes_mask = self.get_observed_relic_nodes_mask(obs)
        team_points = self.get_team_points(obs)
        
        # Log the state
        self.log_state(step, obs)
        
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        if self.player == "player_0":
            logging.info(f"Available Unit IDs: {available_unit_ids}")
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
                if self.player == "player_0":
                    logging.info(f"Discovered new relic node at position {observed_relic_node_positions[id]}")

        
        
        
        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            unit_energy = unit_energys[unit_id]
            if len(self.relic_node_positions) > 0:
                nearest_relic_node_position = self.relic_node_positions[0]
                manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + abs(unit_pos[1] - nearest_relic_node_position[1])
                
                # if close to the relic node we want to hover around it and hope to gain points
                if manhattan_distance <= 4:
                    random_direction = np.random.randint(0, 5)
                    actions[unit_id] = [random_direction, 0, 0]
                else:
                    # otherwise we want to move towards the relic node
                    actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
            else:
                # randomly explore by picking a random location on the map and moving there for about 20 steps
                if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                    rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                    self.unit_explore_locations[unit_id] = rand_loc
                actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]

        # Log the actions
        if self.player == "player_0":
            logging.info(f"Actions: {actions}")

        return actions