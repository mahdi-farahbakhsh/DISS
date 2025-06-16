from abc import ABC, abstractmethod
import numpy as np
import math
import torch


__SEARCH_METHOD__ = {}


def register_search_method(name: str):
    def wrapper(cls):
        if __SEARCH_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __SEARCH_METHOD__[name] = cls
        return cls
    return wrapper


def get_search_algo(name: str, **kwargs):
    if __SEARCH_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SEARCH_METHOD__[name](**kwargs)


class Search(ABC):
    """
    Abstract base class for search-based guidance strategies in sampling.

    Subclasses must implement the `search` method to define how particle indices
    are selected based on reward values at each step.

    Args:
        num_particles (int): Number of particles to manage.
    """
    def __init__(
            self, 
            num_particles=2,
            num_steps=100,
            resample_rate=2,
            annealing='linear',
            **kwargs,
        ):
 
        self.num_particles = num_particles
        self.num_steps = num_steps
        self.num_resample_steps = int(num_steps / resample_rate)
        self.resample_rate = resample_rate
        self.annealing = annealing
    
        # create an annealing schedule over the number of resample steps
        if self.annealing == 'constant':
            init_temp = kwargs.get('init_temp', 1)
            self.annealing_schedule = np.ones(self.num_resample_steps) * init_temp
        elif self.annealing == 'linear':
            init_temp = kwargs.get('init_temp', 10)
            final_temp = kwargs.get('final_temp', 0.1)
            self.annealing_schedule = np.linspace(init_temp, final_temp, self.num_resample_steps)
        elif self.annealing == 'exponential':
            init_temp = kwargs.get('init_temp', 10)
            final_temp = kwargs.get('final_temp', 0.1)
            decay_time = kwargs.get('decay_time', 0.3)  # time constant of decay
            t = np.arange(self.num_resample_steps) / self.num_resample_steps
            self.annealing_schedule = final_temp + (init_temp - final_temp) * np.exp(-t / decay_time)
            

    @abstractmethod
    def search(self, rewards: torch.Tensor, step: int, **kwargs) -> np.ndarray:
        """
        Abstract method to select particle indices based on rewards.

        Args:
            rewards: A tensor containing reward values for particles.
            step (int): Current step in the sampling or optimization process.
            **kwargs: Additional keyword arguments for flexibility.

        Returns:
            np.ndarray: Indices of selected particles.
        """
        raise NotImplementedError
    

# old version of ResampleSearch
# @register_search_method('resample')
# class ResampleSearch(Search):
#     """
#     ResampleSearch is a search-based guidance method that selects particles based on
#     their reward scores. It uses a resampling strategy to select particles with higher
#     rewards more frequently.

#     Attributes:
#         num_particles (int): Number of particles to track.
#         num_steps (int): Total number of steps in the sampling process.
#         resample_rate (int): Rate at which to resample particles.
#         annealing (str): Type of annealing schedule to use ('constant', 'linear', 'exponential').
#     """
#     def __init__(self, num_particles: int, num_steps: int, resample_rate: int, annealing: str, **kwargs):
#         super().__init__(num_particles, num_steps, resample_rate, annealing, **kwargs)

#     def search(self, rewards: torch.Tensor, step: int, **kwargs) -> np.ndarray:

#         idxs = np.arange(self.num_particles)
#         print(f"idxs: {idxs}")

#         if step % self.resample_rate != 0 or step == 0:
#             return idxs
#         else:
#             print('resampling')
            
#             temp = self.annealing_schedule[step // self.resample_rate]

#             print('temp:', temp)

#             # normalize the rewards
#             rewards = rewards.cpu().detach().numpy()
#             norm_rewards = (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards) + 1e-8)
#             values = np.exp(norm_rewards / temp)
#             p = values / np.sum(values)

#             print(f"p: {p}")

#             # resample based on values
#             resampled_idxs = np.random.choice(idxs, p=p, size=self.num_particles, replace=True)
#             print(f"resampled_idxs: {resampled_idxs}")
#             return resampled_idxs
        
@register_search_method('resample')
class ResampleSearch(Search):
    """
    ResampleSearch is a search-based guidance method that selects particles based on
    their reward scores. It uses a resampling strategy to select particles with higher
    rewards more frequently.

    Attributes:
        num_particles (int): Number of particles to track.
        num_steps (int): Total number of steps in the sampling process.
        resample_rate (int): Rate at which to resample particles.
        annealing (str): Type of annealing schedule to use ('constant', 'linear', 'exponential').
    """
    def __init__(self, num_particles: int, num_steps: int, resample_rate: int, annealing: str, **kwargs):
        super().__init__(num_particles, num_steps, resample_rate, annealing, **kwargs)

    def search(self, rewards: torch.Tensor, step: int, **kwargs) -> np.ndarray:

        batch_size = rewards.shape[0]
        resampled_idxs = np.arange(batch_size)
        print(f"resampled_idxs: {resampled_idxs}")

        if step % self.resample_rate != 0 or step == 0:
            return resampled_idxs
        else:
            print('resampling')
            
            temp = self.annealing_schedule[step // self.resample_rate]

            print('temp:', temp)

            # normalize the rewards
            rewards = rewards.cpu().detach().numpy()
            norm_rewards = np.zeros_like(rewards)

            values = np.zeros(batch_size)
            

            for i in range(0, len(rewards), self.num_particles):
                norm_rewards[i: i + self.num_particles] = (rewards[i: i + self.num_particles] - np.max(rewards[i: i + self.num_particles]))

                values[i: i + self.num_particles] = np.exp(norm_rewards[i: i + self.num_particles] / temp)
                p = values[i: i + self.num_particles]  / np.sum(values[i: i + self.num_particles])

                print(f"p: {p}")
                # resample based on values
                resampled_idxs[i: i + self.num_particles] = np.random.choice(np.arange(i, i + self.num_particles),
                                                                              p=p, size=self.num_particles, replace=True)

            print(f"resampled_idxs: {resampled_idxs}")
            return resampled_idxs


@register_search_method('group-meeting')
class GroupMeetingSearch(Search):
    """
    GroupMeetingSearch is a search-based guidance method that selects particles based on
    reward scores within local groups. It uses a hierarchical grouping strategy controlled
    by `base` and `min_group` to decide group sizes dynamically over time.

    Attributes:
        num_particles (int): Number of particles to track.
        base (int): Base step size to determine when resampling occurs.
        min_group (int): Minimum group size used in group-based resampling.
    """
    def __init__(self, num_particles: int, num_steps: int, resample_rate: int, annealing: str, **kwargs):
        """
        Initializes the GroupMeetingSearch object.

        Args:
            num_particles (int): Number of particles to maintain. Must be a power of 2.
            base (int): Base frequency at which resampling groups increase in size.
            min_group (int): Minimum size of each group.
        """
        super().__init__(num_particles, num_steps, resample_rate, annealing, **kwargs)
        if num_particles & (num_particles - 1):
            raise ValueError('num_particles must be a power of 2')
        self.min_group = kwargs.get('min_group', 2)


    def search(self, rewards: torch.Tensor, step: int, **kwargs) -> np.ndarray:
        """
        Performs group-wise resampling of particle indices based on their reward scores.

        Args:
            rewards (torch.Tensor): Reward values for each particle at the current step.
            step (int): Current time step in the sampling process.
            **kwargs: Additional unused keyword arguments.

        Returns:
            np.ndarray: Array of selected particle indices after resampling.
        """
        batch_size = rewards.shape[0]
        resampled_idxs = np.arange(batch_size)
        print(f"resampled_idxs: {resampled_idxs}")

        if step % self.resample_rate != 0 or step == 0:
            return resampled_idxs
        else:
            print('resampling')
            
            temp = self.annealing_schedule[step // self.resample_rate]

            print('temp:', temp)

            # normalize the rewards
            rewards = rewards.cpu().detach().numpy()
            norm_rewards = np.zeros_like(rewards)

            values = np.zeros(batch_size)

            group_size = min(((step // self.resample_rate) & -(step // self.resample_rate)) * self.min_group, self.num_particles)  # max_group_size is num_particles
            

            for i in range(0, len(rewards), group_size):


                norm_rewards[i: i + group_size] = (rewards[i: i + group_size] - np.max(rewards[i: i + group_size]))

                values[i: i + group_size] = np.exp(norm_rewards[i: i + group_size] / temp)
                p = values[i: i + group_size]  / np.sum(values[i: i + group_size])

                print(f"p: {p}")
                # resample based on values
                resampled_idxs[i: i + group_size] = np.random.choice(np.arange(i, i + group_size),
                                                                              p=p, size=group_size, replace=True)

            print(f"resampled_idxs: {resampled_idxs}")
            return resampled_idxs


        rewards = rewards.cpu().detach().numpy()
        normalized_distances = 1 - (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards) + 1e-8)
        scores = np.exp(-self.normalizing_factor * normalized_distances ** 2)

        group_size = min(((step // self.base) & -(step // self.base)) * self.min_group, len(scores))
        selected_idxs = np.zeros(self.num_particles, dtype=int)

        for i in range(0, self.num_particles, group_size):
            scores[i: i + group_size] = scores[i: i + group_size] / np.sum(scores[i: i + group_size])
            selected_idxs[i: i + group_size] = np.random.choice(
                list(range(i, i + group_size)), p=scores[i: i + group_size], size=group_size, replace=True
            )
        return selected_idxs.astype(int)


# # prev version
# @register_search_method('group-meeting')
# class GroupMeetingSearch(Search):
#     """
#     GroupMeetingSearch is a search-based guidance method that selects particles based on
#     reward scores within local groups. It uses a hierarchical grouping strategy controlled
#     by `base` and `min_group` to decide group sizes dynamically over time.

#     Attributes:
#         num_particles (int): Number of particles to track.
#         base (int): Base step size to determine when resampling occurs.
#         min_group (int): Minimum group size used in group-based resampling.
#     """
#     def __init__(self, num_particles: int, base: int, min_group: int, normalizing_factor=100):
#         """
#         Initializes the GroupMeetingSearch object.

#         Args:
#             num_particles (int): Number of particles to maintain. Must be a power of 2.
#             base (int): Base frequency at which resampling groups increase in size.
#             min_group (int): Minimum size of each group.
#         """
#         super().__init__(num_particles)
#         if num_particles & (num_particles - 1):
#             raise ValueError('num_particles must be a power of 2')
#         self.base = base
#         self.min_group = min_group
#         self.normalizing_factor = normalizing_factor

#     def search(self, rewards: torch.Tensor, step: int, **kwargs) -> np.ndarray:
#         """
#         Performs group-wise resampling of particle indices based on their reward scores.

#         Args:
#             rewards (torch.Tensor): Reward values for each particle at the current step.
#             step (int): Current time step in the sampling process.
#             **kwargs: Additional unused keyword arguments.

#         Returns:
#             np.ndarray: Array of selected particle indices after resampling.
#         """
#         if step % self.base != 0 or step == 0:
#             return np.arange(self.num_particles)

#         rewards = rewards.cpu().detach().numpy()
#         normalized_distances = 1 - (rewards - np.min(rewards)) / (np.max(rewards) - np.min(rewards) + 1e-8)
#         scores = np.exp(-self.normalizing_factor * normalized_distances ** 2)

#         group_size = min(((step // self.base) & -(step // self.base)) * self.min_group, len(scores))
#         selected_idxs = np.zeros(self.num_particles, dtype=int)

#         for i in range(0, self.num_particles, group_size):
#             scores[i: i + group_size] = scores[i: i + group_size] / np.sum(scores[i: i + group_size])
#             selected_idxs[i: i + group_size] = np.random.choice(
#                 list(range(i, i + group_size)), p=scores[i: i + group_size], size=group_size, replace=True
#             )
#         return selected_idxs.astype(int)


@register_search_method('FK-steering')
class FKSteeringSearch(GroupMeetingSearch):
    def __init__(self, num_particles, base, method='max'):
        super().__init__(num_particles=num_particles, base=base, min_group=num_particles)
        self.method = method
        self.history_rewards = np.zeros(self.num_particles)
        if method == 'max':
            self.history_rewards = -1e6 * np.ones(self.num_particles)

    def search(self, rewards, step, **kwargs):
        rewards = - np.array(rewards) ** 2
        if self.method == 'difference':
            p = rewards - self.history_rewards
            self.history_rewards = rewards
        elif self.method == 'max':
            if (step - 1) % int(self.base * self.num_particles / 2) == 0:
                self.history_rewards = -1e6 * np.ones(self.num_particles)
            self.history_rewards = np.maximum(rewards, self.history_rewards)
            p = self.history_rewards
            print(np.sqrt(-p))
        elif self.method == 'sum':
            if (step - 1) % int(self.base * self.num_particles / 2) == 0:
                self.history_rewards = np.zeros(self.num_particles)
            self.history_rewards += rewards
            p = self.history_rewards
            print(np.sqrt(-p))
        else:
            raise ValueError(f"Method {self.method} not supported!")

        p = np.exp(600 * p / np.max(np.abs(p)))

        p = p / np.sum(p)
        selected_idxs = np.random.choice(list(range(self.num_particles)), p=p, size=self.num_particles, replace=True)
        self.history_rewards = self.history_rewards[selected_idxs]

        return selected_idxs
