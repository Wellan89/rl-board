from gym.envs.registration import register

register(
    id='csb-v0',
    entry_point='envs.csb:CsbEnv',
)
