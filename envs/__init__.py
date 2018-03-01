from gym.envs.registration import register

register(
    id='csb-d0-v0',
    entry_point='envs.csb.env:CsbEnvV0D0',
)
register(
    id='csb-d1-v0',
    entry_point='envs.csb.env:CsbEnvV0D1',
)
