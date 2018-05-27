from gym.envs.registration import register


register(
    id='csb-d0-v1',
    entry_point='envs.csb.csb_env:CsbEnvD0',
)
register(
    id='csb-d1-v1',
    entry_point='envs.csb.csb_env:CsbEnvD1',
)
register(
    id='csb-d2-v1',
    entry_point='envs.csb.csb_env:CsbEnvD2',
)
register(
    id='csb-d3-v1',
    entry_point='envs.csb.csb_env:CsbEnvD3',
)

register(
    id='stc-d0-v1',
    entry_point='envs.stc.stc_env:StcEnvD0',
)
register(
    id='stc-d1-v1',
    entry_point='envs.stc.stc_env:StcEnvD1',
)
register(
    id='stc-d2-v1',
    entry_point='envs.stc.stc_env:StcEnvD2',
)
register(
    id='stc-d3-v1',
    entry_point='envs.stc.stc_env:StcEnvD3',
)
