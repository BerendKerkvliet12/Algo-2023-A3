import math
import numpy as np
import typing


class DroneExtinguisher:
    def __init__(self, forest_location: typing.Tuple[float, float], bags: typing.List[int], 
                 bag_locations: typing.List[typing.Tuple[float, float]], 
                 liter_cost_per_km: float, liter_budget_per_day: int, usage_cost: np.ndarray):
        """
        The DroneExtinguisher object. This object contains all functions necessary to compute the most optimal way of saving the forest
        from the fire using dynamic programming. Note that all costs that we use in this object will be measured in liters. 

        :param forest_location: the location (x,y) of the forest 
        :param bags: list of the contents of the water bags in liters
        :param bag_locations: list of the locations of the water bags
        :param liter_cost_per_km: the cost of traveling a kilometer with drones, measured in liters of waters 
        :param liter_budget_per_day: the maximum amount of work (in liters) that we can do per day 
                                     (sum of liter contents transported on the day + travel cost in liters)
        :param usage_cost: a 2D array. usage_cost[i,k] is the cost of flying water bag i with drone k from the water bag location to the forest
        """

        self.forest_location = forest_location
        self.bags = bags
        self.bag_locations = bag_locations
        self.liter_cost_per_km = liter_cost_per_km
        self.liter_budget_per_day = liter_budget_per_day
        self.usage_cost = usage_cost # usage_cost[i,k] = additional cost to use drone k to for bag i

        # the number of bags and drones that we have in the problem
        self.num_bags = len(self.bags)
        self.num_drones = self.usage_cost.shape[1] if not usage_cost is None else 1

        # list of the travel costs measured in the amount of liters of water
        # that could have been emptied in the forest (measured in integers)
        self.travel_costs_in_liters = []

        # idle_cost[i,j] is the amount of time measured in liters that we are idle on a day if we 
        # decide to empty bags[i:j+1] on that day
        self.idle_cost = -1*np.ones((self.num_bags, self.num_bags))

        # optimal_cost[i,k] is the optimal cost of emptying water bags[:i] with drones[:k+1]
        # this has to be filled in using the dynamic programming function
        self.optimal_cost = np.zeros((self.num_bags + 1, self.num_drones))

        # Data structure that can be used for the backtracing method (NOT backtracking):
        # reconstructing what bags we empty on every day in the forest
        self.backtrace_memory = dict()


        #######################################################remove if unneeded
        #extra code needed for dynamic programming
        # self.dp = None
        # self.liters_used = None  
    
    @staticmethod
    def compute_euclidean_distance(point1: typing.Tuple[float, float], point2: typing.Tuple[float, float]) -> float:
        """
        A static method (as it does not have access to the self. object) that computes the Euclidean
        distance between two points

        :param point1: an (x,y) tuple indicating the location of point 1
        :param point2: idem for point2

        Returns 
          float: the Euclidean distance between the two points
        """
        
        # # TODO
        # raise NotImplementedError()

        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def fill_travel_costs_in_liters(self):
        """
        Function that fills in the self.travel_costs_in_liters data structure such that
        self.travel_costs_in_liters[i] is the cost of traveling from the forest/drone housing
        to the bag AND back to the forest, measured in liters of waters (using liter_cost_per_km)
        Note: the cost in liters should be rounded up (with, e.g., np.ceil)
                
        The function does not return anything.  
        """
        
        # # TODO
        # raise NotImplementedError()
           # Iterate over all bag locations
        for bag_location in self.bag_locations:
            # Calculate the Euclidean distance from the forest to the bag location
            distance = self.compute_euclidean_distance(self.forest_location, bag_location)

            # Calculate the cost in liters for the round trip (to the bag location and back)
            # We multiply the distance by 2 (for the round trip) and by the liter cost per kilometer
            # We use np.ceil to round up the result
            liter_cost = np.ceil(2 * distance * self.liter_cost_per_km)

            # Add the calculated cost to the list of travel costs
            self.travel_costs_in_liters.append(liter_cost)

            

    def compute_sequence_idle_time_in_liters(self, i, j):
        """
        Function that computes the idle time (time not spent traveling to/from bags or emptying bags in the forest)
        in terms of liters. This function assumes that self.travel_costs_in_liters has already been filled with the
        correct values using the function above, as it makes use of that data structure.
        More specifically, this function computes the idle time on a day if we decide to empty self.bags[i:j+1] 
        (bag 0, bag 1, ..., bag j) on that day.

        Note: the returned idle time can be negative (if transporting the bags is not possible within a day) 

        :param i: integer index 
        :param j: integer index

        Returns:
          int: the amount of time (measured in liters) that we are idle on the day   
        """
        
        # # TODO
        # raise NotImplementedError()
        
        # Calculate the total cost of transporting and emptying the bags from i to j
        # This includes the travel cost (round trip to each bag location and back) and the water content of each bag
        total_cost = sum(self.travel_costs_in_liters[i:j+1]) + sum(self.bags[i:j+1])

        # The idle time is the difference between the liter budget per day and the total cost
        # If the total cost exceeds the liter budget per day, the idle time will be negative
        idle_time = self.liter_budget_per_day - total_cost

        return idle_time

    def compute_idle_cost(self, i, j, idle_time_in_liters):
        """
        Function that transforms the amount of time that we are idle on a day if we empty self.bags[i:j+1]
        on a day (idle_time_in_liters) into a quantity that we want to directly optimize using the formula
        in the assignment description. 
        If transporting self.bags[i:j+1] is not possible within a day, we should return np.inf as cost. 
        Moreover, if self.bags[i:j+1] are the last bags that are transported on the final day, the idle cost is 0 
        as the operation has been completed. In all other cases, we use the formula from the assignment text. 

        You may not need to use every argument of this function

        :param i: integer index
        :param j: integer index
        :param idle_time_in_liters: the amount of time that we are idle on a day measured in liters

        Returns
          - integer: the cost of being idle on a day corresponding to idle_time_in_liters
        """
        
        # # TODO
        # raise NotImplementedError()

        
        # If the idle time is negative, it means that transporting the bags is not possible within a day
        # In this case, we return np.inf as cost
        if idle_time_in_liters < 0:
            return np.inf

        # If the bags are the last ones that are transported on the final day, the idle cost is 0
        # as the operation has been completed
        if j == self.num_bags - 1:
            return 0

        # In all other cases, we calculate the idle cost using the formula from the assignment text
        idle_cost = idle_time_in_liters**3

        return idle_cost


    def compute_sequence_usage_cost(self, i: int, j: int, k: int) -> float:
        """
        Function that computes and returns the cost of using drone k for self.bags[i:j+1], making use of
        self.usage_cost, which gives the cost for every bag-drone pair. 
        Note: the usage cost is independent of the distance to the forest. This is purely the operational cost
        to use drone k for bags[i:j+1].

        :param i: integer index
        :param j: integer index
        :param k: integer index

        Returns
          - float: the cost of usign drone k for bags[i:j+1] 
        """
        
        # # TODO
        # raise NotImplementedError()

        # Initialize the total usage cost to 0
        total_usage_cost = 0

        # Iterate over the bags from i to j
        for bag_index in range(i, j+1):
            # Add the usage cost of using drone k for the current bag to the total usage cost
            total_usage_cost += self.usage_cost[bag_index, k]

        return total_usage_cost


    def dynamic_programming(self):
        """
        The function that uses dynamic programming to solve the problem: compute the optimal way of emptying bags in the forest
        per day and store a solution that can be used in the backtracing function below (if you want to do that assignment part). 
        In this function, we fill the memory structures self.idle_cost and self.optimal_cost making use of functions defined above. 
        This function does not return anything. 
        """
        
        # # TODO
        # raise NotImplementedError()
        
        num_bags = len(self.bags)
        num_drones = len(self.usage_cost[0])
    
        # Initialize the DP table
        self.dp = [[float('inf')] * (num_drones + 1) for _ in range(num_bags + 1)]
        self.dp[0] = [0] * (num_drones + 1)
    
        # Fill the DP table
        for i in range(1, num_bags + 1):
            for k in range(1, num_drones + 1):
                for j in range(i):
                    # Calculate the cost of using drone k to transport water bags from j to i
                    cost = sum(self.usage_cost[bag][k - 1] for bag in range(j, i)) + self.travel_costs_in_liters[i - 1]
                    # Update the DP table
                    self.dp[i][k] = min(self.dp[i][k], self.dp[j][k - 1] + cost)



        # ## to work with def test_dynamic_programming_simple(self):
        # # Initialize the optimal cost for the first bag with each drone
        # for k in range(self.num_drones):
        #     self.optimal_cost[1, k] = self.compute_sequence_usage_cost(0, 0, k)

        # # Iterate over the rest of the bags
        # for i in range(2, self.num_bags + 1):
        #     # Iterate over the drones
        #     for k in range(self.num_drones):
        #         # Initialize the minimum cost as infinity
        #         min_cost = np.inf

        #         # Iterate over the possible starting points for the bags
        #         for j in range(i):
        #             # Compute the idle time in liters
        #             idle_time_in_liters = self.compute_sequence_idle_time_in_liters(j, i - 1)

        #             # Compute the idle cost
        #             idle_cost = self.compute_idle_cost(j, i - 1, idle_time_in_liters)

        #             # Compute the usage cost
        #             usage_cost = self.compute_sequence_usage_cost(j, i - 1, k)

        #             # Compute the total cost
        #             total_cost = idle_cost + usage_cost

        #             # If the total cost is less than the minimum cost, update the minimum cost
        #             if total_cost < min_cost:
        #                 min_cost = total_cost

        #         # Update the optimal cost for the current bag with the current drone
        #         self.optimal_cost[i, k] = min_cost


        # ###to work with def test_dyanmic_programming_one_day(self):
        ## no code does already work with this test

        # # Initialize the optimal cost array with infinity
        # self.optimal_cost = [float('inf')] * len(self.bags)
        # self.optimal_cost[0] = 0



        # # Iterate over all bags
        # for i in range(len(self.bags)):
        #     # Iterate over all drones
        #     for k in range(len(self.usage_cost[0])):
        #         # Calculate the cost of using drone k for bags[i:j+1]
        #         for j in range(i, len(self.bags)):
        #             cost = self.compute_sequence_usage_cost(i, j, k)
        #             # Update the optimal cost if the new cost is lower
        #             if cost < self.optimal_cost[j]:
        #                 self.optimal_cost[j] = cost

        # ###to work with def test_dyanmic_programming_no_travel_cost(self):
        # # Initialize the optimal_cost array with infinity
        # self.optimal_cost = [[float('inf')] * (len(self.bags) + 1) for _ in range(len(self.bags) + 1)]

        # # The cost of using the first drone for the first bag is just the usage cost
        # self.optimal_cost[0][1] = self.usage_cost[0][0] if self.usage_cost.shape[1] > 0 else 0

        # # Iterate over all bags
        # for j in range(1, len(self.bags)):
        #     # Iterate over all drones
        #     for k in range(j + 1):
        #         # Compute the minimum cost for using the k-th drone for the j-th bag
        #         if k == 0:
        #             self.optimal_cost[k][j] = self.optimal_cost[k][j - 1] + (self.compute_sequence_usage_cost(k, j, k) if self.usage_cost.shape[1] > k else 0)
        #         else:
        #             self.optimal_cost[k][j] = min(self.optimal_cost[i][j - 1] + (self.compute_sequence_usage_cost(i, j, k) if self.usage_cost.shape[1] > k else 0) for i in range(k))

        
        # ###to work with def test_dyanmic_programming_with_travel_cost(self):
        # # Initialize the cost matrix with infinity
        # self.optimal_cost = np.full((len(self.bags), self.liter_budget_per_day + 1), np.inf)

        # # The cost of not using any bag is 0
        # self.optimal_cost[0, :] = 0

        # # Iterate over all bags
        # for i in range(1, len(self.bags)):
        #     # Iterate over all possible liter budgets
        #     for j in range(self.liter_budget_per_day + 1):
        #         # If the bag can be used within the current liter budget
        #         if j >= self.travel_costs_in_liters[i]:
        #             # Compute the cost of using the bag
        #             use_bag_cost = self.usage_cost[i] + self.optimal_cost[i - 1, j - self.travel_costs_in_liters[i]]
        #             # Compute the cost of not using the bag
        #             not_use_bag_cost = self.optimal_cost[i - 1, j]
        #             # Choose the minimum cost
        #             self.optimal_cost[i, j] = min(use_bag_cost, not_use_bag_cost)
        #         else:
        #             # If the bag cannot be used within the current liter budget, do not use the bag
        #             self.optimal_cost[i, j] = self.optimal_cost[i - 1, j]

        # # The lowest cost is the minimum cost of using all bags
        # self.lowest_cost = np.min(self.optimal_cost[-1, :])
        
        
        # ###to work with def test_dyanmic_programming_with_travel_cost_multiple_drones(self):
    #    # Initialize the cost matrix with infinity
    #     self.optimal_cost = np.full((self.num_bags + 1, self.liter_budget_per_day + 1), np.inf)
    #     self.optimal_cost[0, :] = 0

    #     # Iterate over all bags
    #     for i in range(1, self.num_bags + 1):
    #         # Iterate over all possible liter budgets
    #         for j in range(self.liter_budget_per_day + 1):
    #             # Iterate over all possible number of liters to be used for the current bag
    #             for k in range(min(j, self.bags[i - 1]) + 1):
    #                 # Calculate the cost of using k liters for the current bag
    #                 cost = self.usage_cost[i - 1, k] + self.idle_cost[i - 1, j - k]
    #                 # Update the optimal cost if the current cost is lower
    #                 if cost < self.optimal_cost[i, j]:
    #                     self.optimal_cost[i, j] = cost
    #                     self.liters_used[i, j] = k

    #     # Backtrace to find the optimal number of liters to be used for each bag
    #     remaining_liters = self.liter_budget_per_day
    #     for i in range(self.num_bags, 0, -1):
    #         self.optimal_liters[i - 1] = self.liters_used[i, remaining_liters]
    #         remaining_liters -= self.optimal_liters[i - 1]


    def lowest_cost(self) -> float:
        """
        Returns the lowest cost at which we can empty the water bags to extinguish to forest fire. Inside of this function,
        you can assume that self.dynamic_progrmaming() has been called so that in this function, you can simply extract and return
        the answer from the filled in memory structure.

        Returns:
          - float: the lowest cost
        """
        
        # # TODO
        # raise NotImplementedError()
        return np.min(self.optimal_cost)

        # The lowest cost is the last element in the optimal cost array
        return self.optimal_cost[-1]

        # # # The lowest cost is the minimum cost of using any drone for the last bag
        # return min(self.optimal_cost[k][-1] for k in range(len(self.bags)))



    def backtrace_solution(self) -> typing.List[int]:
        """
        Returns the solution of how the lowest cost was obtained by using, for example, self.backtrace_memory (but feel free to do it your own way). 
        The solution is a tuple (leftmost indices, drone list) as described in the assignment text. Here, leftmost indices is a list 
        [idx(1), idx(2), ..., idx(T)] where idx(i) is the index of the water bag that is emptied left-most (at the start of the day) on day i. 
        Drone list is a list [d(0), d(1), ..., d(num_bags-1)] where d(j) tells us which drone was used in the optimal
        solution to transport water bag j.  
        See the assignment description for an example solution. 

        This function does not have to be made - you can still pass the assignment if you do not hand this in,
        however it will cost a full point if you do not do this (and the corresponding question in the report).  
            
        :return: A tuple (leftmost indices, drone list) as described above
        """
        
        # # TODO
        # raise NotImplementedError()

        # # # Backtrace to find the optimal number of liters to be used for each bag
        # # remaining_liters = self.liter_budget_per_day
        # # for i in range(self.num_bags, 0, -1):
        # #     self.optimal_liters[i - 1] = self.liters_used[i, remaining_liters]
        # #     remaining_liters -= self.optimal_liters[i - 1]


        #  # Initialize the lists for the leftmost indices and the drone list
        # leftmost_indices = []
        # drone_list = []
    
        # # Initialize the remaining liters with the total liter budget per day
        # remaining_liters = self.liter_budget_per_day
    
        # # Backtrace the optimal solution
        # for i in range(self.num_bags, 0, -1):
        #     # Find the number of liters used for the current bag
        #     liters_used = self.liters_used[i, remaining_liters]
    
        #     # Find the drone that was used for the current bag
        #     drone = np.argmin(self.usage_cost[i - 1, :])
    
        #     # Add the index of the current bag to the leftmost indices list
        #     leftmost_indices.append(i - 1)
    
        #     # Add the drone to the drone list
        #     drone_list.append(drone)
    
        #     # Update the remaining liters
        #     remaining_liters -= liters_used
    
        # # Reverse the lists to get the correct order
        # leftmost_indices.reverse()
        # drone_list.reverse()
    
        # return leftmost_indices, drone_list
