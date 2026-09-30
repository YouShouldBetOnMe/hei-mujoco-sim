"""Shared schema without simulator dependencies; importable in training env."""
CAMERAS = ('front', 'left_wrist', 'right_wrist')
STATE_NAMES = sum(([*[f'{side}_joint_{j}.pos' for j in range(1, 7)],
                    f'{side}_gripper.pos'] for side in ('right', 'left')), [])
STATE_NAMES += ['x.vel', 'y.vel', 'theta.vel', 'height.pos']
