from gym.envs.registration import register


register(
    id='csb-d0-v1',
    entry_point='envs.csb.env:CsbEnvD0V1',
)
register(
    id='csb-d1-v1',
    entry_point='envs.csb.env:CsbEnvD1V1',
)
register(
    id='csb-versus-v1',
    entry_point='envs.csb.env:CsbEnvVersusV1',
)
