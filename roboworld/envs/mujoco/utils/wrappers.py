import gym


class DataCollectionWrapper(gym.Wrapper):
    def __init__(self, env, record_images=True):
        super(DataCollectionWrapper, self).__init__(env)
        self.record_images = record_images
        self.data = {
            "observations": [],
            "actions": [],
            "images": [],
            "dones": [],
        }

    def step(self, action):
        # Step the environment
        obs, reward, done, info = self.env.step(action)

        # Record action and obs
        self.data["actions"].append(action)
        self.data["observations"].append(obs)
        self.data["dones"].append(done)

        return obs, reward, done, info

    def reset(self, **kwargs):
        # Clear data storage when resetting the environment
        self.data = {
            "observations": [],
            "actions": [],
            "images": [],
            "dones": [],
        }
        return self.env.reset(**kwargs)

    def get_collected_data(self):
        return self.data
