VIEWPORT_W = 710
VIEWPORT_H = 400

BLOCK_SIZE = 30

BLOCK_COLORS = [
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    None,
    (0.0, 0.0, 0.0)
]


def render(world, viewer=None, mode='human'):
    from gym.envs.classic_control import rendering

    if viewer is None:
        viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

    def _draw_grid(grid, offset):
        for i, row in enumerate(grid[::-1]):
            for j, block in enumerate(row):
                color = BLOCK_COLORS[block]
                if color:
                    viewer.draw_circle(color=color, radius=BLOCK_SIZE * 0.5).add_attr(
                        rendering.Transform(translation=(offset[0] + (j + 0.5) * BLOCK_SIZE,
                                                         offset[1] + (i + 0.5) * BLOCK_SIZE))
                    )

    state = world.compute_state(False)
    next_blocks = [(state[2 * i], state[2 * i + 1]) for i in range(8)]
    player_1_grid = [state[(16 + 6 * i):(16 + 6 * (i + 1))] for i in range(12)]
    player_2_grid = [state[(16 + 12 * 6 + 6 * i):(16 + 12 * 6 + 6 * (i + 1))] for i in range(12)]

    _draw_grid(player_1_grid, offset=(0.0, 0.0))
    _draw_grid(next_blocks[::-1], offset=((VIEWPORT_W - BLOCK_SIZE * 2.0) * 0.5, 0.0))
    _draw_grid(player_2_grid, offset=(VIEWPORT_W - BLOCK_SIZE * 6.0, 0.0))

    return viewer, viewer.render(return_rgb_array=(mode == 'rgb_array'))
