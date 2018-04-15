from gym.envs.registration import register


register(
    id='csb-v1',
    entry_point='envs.csb.env:CsbEnv',
)
