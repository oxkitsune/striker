class Entity:
    def __init__(self, scene, init_pos, init_yaw, radius: float):
        self.scene = scene
        self.init_pos = init_pos
        self.init_yaw = init_yaw
        self.radius = radius
