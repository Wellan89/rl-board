import math

from envs.csb.vincent_algo import Pos

VIEWPORT_W = 710
VIEWPORT_H = 400


def render(world, viewer=None, mode='human'):
    from gym.envs.classic_control import rendering

    if viewer is None:
        viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)

    def _pos_to_screen(_p):
        return _p.x * VIEWPORT_W / 16000, _p.y * VIEWPORT_H / 9000

    def _radius_to_screen(_r):
        return _r * VIEWPORT_W / 16000

    cp_radius = _radius_to_screen(world.circuit.cp(0).r + world.pods[0].r)
    for i in range(world.circuit.nbcp()):
        cp = world.circuit.cp(i)
        color_ratio = cp.id / (world.circuit.nbcp() - 1)
        color = (color_ratio * 0.8, color_ratio * 0.8, 0.2 + color_ratio * 0.8)
        viewer.draw_circle(color=color, radius=cp_radius).add_attr(
            rendering.Transform(translation=_pos_to_screen(cp))
        )

    pod_radius = _radius_to_screen(world.pods[0].r)
    for pod in world.pods:
        color = (float(pod.id >= 2), float(pod.id < 2), 0.0)
        if pod.shield > 0:
            viewer.draw_circle(color=(0.0, 0.0, 0.6),
                               radius=pod_radius + _radius_to_screen(20 * pod.shield)).add_attr(
                rendering.Transform(translation=_pos_to_screen(pod))
            )
        viewer.draw_circle(color=color, radius=pod_radius).add_attr(
            rendering.Transform(translation=_pos_to_screen(pod))
        )
        if pod.boost_available:
            viewer.draw_circle(color=(0.0, 0.0, 0.0),
                               radius=_radius_to_screen(80)).add_attr(
                rendering.Transform(translation=_pos_to_screen(pod))
            )
        viewer.draw_line(
            color=color,
            start=_pos_to_screen(pod),
            end=_pos_to_screen(pod.next_checkpoint()),
        )
        pod_angle_rad = pod.angle * math.pi / 180.0
        viewer.draw_line(
            color=(0.0, 0.0, 0.0),
            start=_pos_to_screen(pod),
            end=_pos_to_screen(Pos(pod.x + 300.0 * math.cos(pod_angle_rad),
                                   pod.y + 300.0 * math.sin(pod_angle_rad))),
        )

    return viewer, viewer.render(return_rgb_array=(mode == 'rgb_array'))
