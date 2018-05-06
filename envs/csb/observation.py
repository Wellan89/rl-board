def observation(world, pods):
    features = [float(world.nblaps), float(world.circuit.nbcp())]
    for pod in pods:
        features += [
            pod.x / 5000.0,
            pod.y / 5000.0,
            pod.vx / 5000.0,
            pod.vy / 5000.0,
            pod.angle / 360.0,
            float(pod.boost_available),
            pod.timeout / 100.0,
            pod.shield / 4.0,
            float(pod.lap),
            float(pod.ncpid),
        ]
        for i in range(3):
            cp = pod.next_checkpoint(i)
            features += [
                cp.x / 5000.0,
                cp.y / 5000.0,
            ]
    return features
