from gym.envs.registration import register

register(
    id='csb-d0-v0',
    entry_point='envs.csb.env:CsbEnvD0V0',
)
register(
    id='csb-d1-v0',
    entry_point='envs.csb.env:CsbEnvD1V0',
)
