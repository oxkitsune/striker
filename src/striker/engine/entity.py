class Entity:
    def __init__(
        self,
        scene,
        init_pos,
        init_yaw,
        init_vel,
        radius: float,
        mass: float = 1.0,
        restitution: float = 0.9,
    ):
        self.scene = scene
        self.init_pos = init_pos
        self.init_yaw = init_yaw
        self.init_vel = init_vel
        self.radius = radius
        self.mass = mass
        self.restitution = restitution
