from gym.envs.registration import register


register(
    id='csb-d0-v1',
    entry_point='envs.csb.env:CsbEnvD0',
)
register(
    id='csb-d1-v1',
    entry_point='envs.csb.env:CsbEnvD1',
)
register(
    id='csb-d2-v1',
    entry_point='envs.csb.env:CsbEnvD2',
)

