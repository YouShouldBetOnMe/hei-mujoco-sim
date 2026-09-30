"""Dynamic tabletop simulation. This module never imports a hardware client."""
from pathlib import Path
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile

import mujoco
import numpy as np

from .schema import CAMERAS, STATE_NAMES
PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = Path.cwd()


def model_source():
    source = PACKAGE_ROOT / 'assets/mujoco_ik'
    if not (source / 'model/HEI_robot_urdf/urdf/HEI_robot_urdf.urdf').is_file():
        raise FileNotFoundError('Packaged robot assets are missing; reinstall hei-mujoco-sim.')
    if os.name == 'nt':
        # Use a content-addressed ASCII cache for the Windows URDF loader.
        # Different installs/model revisions never overwrite each other's cache.
        digest = hashlib.sha256()
        for file in sorted(source.rglob('*')):
            if file.is_file() and '__pycache__' not in file.parts:
                digest.update(file.relative_to(source).as_posix().encode())
                digest.update(file.read_bytes())
        target = Path(tempfile.gettempdir()) / 'hei_mujoco_sim' / digest.hexdigest()[:20]
        shutil.copytree(source, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        source = target
    return source


def scene_builder(directory):
    name = '_hei_packaged_scene_' + hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, directory / 'hei_robot_mujoco_scene.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def camera_axes(position, target, up=(0, 0, 1)):
    z = np.array(position, float) - np.array(target, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.r_[x, y]


class Environment:
    fps = 10
    size = 224

    def __init__(self, size=224, fixed_lift=None):
        self.size = size
        self.fixed_lift = fixed_lift
        directory = model_source()
        scene = scene_builder(directory)
        self.objects = scene.GRASPABLE_OBJECTS
        spec = mujoco.MjSpec.from_file(str(directory / 'model/HEI_robot_urdf/urdf/HEI_robot_urdf.urdf'))
        if fixed_lift is not None:
            # Remove the lift DOF entirely for fixed-base dataset collection.
            # Apply its constant displacement in the parent's coordinates.
            joint = spec.joint('lift_joint')
            carriage = spec.body('lift_carriage_link')
            displacement = np.zeros(3)
            mujoco.mju_rotVecQuat(displacement, np.asarray(joint.axis)*fixed_lift, carriage.quat)
            carriage.pos = np.asarray(carriage.pos) + displacement
            spec.delete(joint)
        spec.option.timestep = .002
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.iterations = 60
        # Mesh finger/box contacts need a contact patch, not a single point.
        # Elliptic cones and the friction-only NoSlip pass suppress the soft
        # contact model's slow creep while retaining Coulomb force limits.
        spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10
        spec.option.noslip_iterations = 5
        # Separate collision groups: imported robot meshes are not calibrated for
        # self collision; robot/object/environment contacts remain active.
        for geom in spec.geoms:
            if geom.contype:
                geom.contype, geom.conaffinity = 2, 1
                geom.friction = [1.0, .01, .001]
                geom.condim = 6
        scene._add_floor_and_sky(spec)
        scene._add_lights(spec)
        scene._add_table(spec)
        scene._add_graspable_objects(spec)
        for geom in spec.geoms:
            if geom.name.startswith(('hei_ground', 'hei_table_', 'hei_object_')):
                geom.contype, geom.conaffinity = 1, 3
                geom.friction = [.8, .01, .001]
                geom.condim = 6
        for obj in self.objects:
            spec.body(obj.body_name).add_freejoint(name=obj.body_name + '_free')
            spec.geom(obj.geom_name).mass = .12 if 'cube' in obj.body_name else .10
        # The external overview camera is for the operator only. It is never
        # included in CAMERAS or returned by observe().
        pos = [1.65, -1.5, 1.65]
        spec.worldbody.add_camera(name='overview', pos=pos,
            xyaxes=camera_axes(pos, [.65, 0, .85]), fovy=58)
        # Front-facing camera on the robot torso/carriage, between and above
        # the arm mounts. These are provisional mounts, not real calibration.
        pos = [.12, 0, .30]
        spec.body('lift_carriage_link').add_camera(name='front', pos=pos,
            xyaxes=camera_axes(pos, [.8, 0, -.10 if fixed_lift is not None else -.20]),
            fovy=75 if fixed_lift is not None else 70)
        for side, prefix in [('left', 'b_left'), ('right', 'a_right')]:
            pos = [.04, 0, .14]
            spec.body(prefix + '_link6').add_camera(name=side+'_wrist', pos=pos,
                xyaxes=camera_axes(pos, [.45, 0, -.42]), fovy=85)
        self.joint_names = []
        for prefix in ('a_right', 'b_left'):
            self.joint_names += [f'{prefix}_joint{i}' for i in range(1, 7)]
            self.joint_names += [f'{prefix}_end_joint_L_finger', f'{prefix}_end_joint_R_finger']
        if fixed_lift is None:
            self.joint_names.append('lift_joint')
        for name in self.joint_names:
            joint = spec.joint(name)
            joint.armature = .02
            actuator = spec.add_actuator(name='servo_' + name,
                target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            finger, lift = 'finger' in name, name == 'lift_joint'
            actuator.set_to_position(kp=500 if finger else 3000 if lift else 180,
                kv=15 if finger else 250 if lift else 20)
            actuator.ctrllimited = True
            actuator.ctrlrange = joint.range
            # Approximate simulation limits, NOT calibrated hardware limits.
            force = 25 if finger else 800 if lift else 80
            actuator.forcelimited = True
            actuator.forcerange = [-force, force]
        self.model = spec.compile()
        self.spec = spec  # Retain the fully configured MJCF for settings export.
        self.data = mujoco.MjData(self.model)
        self.ikdata = mujoco.MjData(self.model)
        self.qadr = np.array([self.model.joint(n).qposadr[0] for n in self.joint_names])
        self.dadr = np.array([self.model.joint(n).dofadr[0] for n in self.joint_names])
        self.renderer = None
        self.target = np.zeros(18, np.float32)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for start in (0, 7):
            self.target[start:start+7] = [0, -.5, -.5, 0, 0, 0, .08]
        self.target[14:] = 0
        if self.fixed_lift is not None:
            self.target[17] = self.fixed_lift
        self._set_ctrl(self.target)
        self.data.qpos[self.qadr] = self.data.ctrl
        mujoco.mj_forward(self.model, self.data)
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0
        mujoco.mj_forward(self.model, self.data)

    def _set_ctrl(self, action):
        action = np.asarray(action)
        if action.shape != (18,) or not np.isfinite(action).all():
            raise ValueError('Action must have 18 finite components')
        if np.any(action[14:17] != 0):
            raise ValueError('This tabletop environment has a fixed chassis')
        values = []
        for i in (0, 7):
            values.extend(action[i:i+6])
            opening = np.clip(action[i+6], 0, .1) / 2
            values.extend([opening, -opening])
        if self.fixed_lift is None:
            values.append(action[17])
        elif abs(float(action[17])-self.fixed_lift)>1e-7:
            raise ValueError('Lift is physically fixed in this environment')
        self.data.ctrl[:] = np.clip(values, self.model.actuator_ctrlrange[:, 0],
                                   self.model.actuator_ctrlrange[:, 1])

    def step(self, action=None):
        if action is not None:
            self.target[:] = action
        self._set_ctrl(self.target)
        for _ in range(round(1 / self.fps / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError('Non-finite simulation state')

    def state(self):
        q = self.data.qpos[self.qadr]
        height = q[16] if self.fixed_lift is None else self.fixed_lift
        return np.array([*q[:6], q[6]-q[7], *q[8:14], q[14]-q[15], 0, 0, 0, height], np.float32)

    def command(self, side, translation, rotation, gripper, lift=0):
        """Damped Jacobian IK for one 0.1 s operator command in world coordinates."""
        self.ikdata.qpos[:] = self.data.qpos
        self._set_ctrl(self.target)
        self.ikdata.qpos[self.qadr] = self.data.ctrl
        mujoco.mj_forward(self.model, self.ikdata)
        body = self.model.body(('a_right' if side == 'right' else 'b_left') + '_tcp').id
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.ikdata, jp, jr, body)
        ai, qi = (0, 0) if side == 'right' else (7, 8)
        columns = self.dadr[qi:qi+6]
        jac = np.vstack([jp[:, columns], .35 * jr[:, columns]])
        delta = np.r_[np.asarray(translation) / self.fps,
                      .35 * np.asarray(rotation) / self.fps]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + .0025*np.eye(6), delta)
        low = self.model.actuator_ctrlrange[qi:qi+6, 0]
        high = self.model.actuator_ctrlrange[qi:qi+6, 1]
        self.target[ai:ai+6] = np.clip(self.target[ai:ai+6] + np.clip(dq, -.07, .07), low, high)
        self.target[ai+6] = np.clip(self.target[ai+6] + gripper / self.fps, 0, .1)
        if self.fixed_lift is not None:
            if lift != 0:
                raise ValueError('Lift motion disabled for fixed-base collection')
            self.target[17] = self.fixed_lift
        else:
            self.target[17] = np.clip(self.target[17] + lift / self.fps, -.8, 0)
        return self.target.copy()

    def observe(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.size, width=self.size)
        timestamp = float(self.data.time)
        frames = {}
        for name in CAMERAS:
            self.renderer.update_scene(self.data, camera=name)
            frames[name] = self.renderer.render().copy()
        if self.data.time != timestamp:
            raise RuntimeError('Physics changed during camera capture')
        return dict(timestamp=timestamp, state=self.state(), images=frames)

    def metadata(self):
        return dict(fps=self.fps, image_size=[self.size, self.size, 3], cameras=list(CAMERAS),
            state_names=STATE_NAMES, action_names=STATE_NAMES, action_semantics='absolute_position_targets',
            units='arm:rad, gripper:total_opening_m, height:m, chassis:m/s,m/s,rad/s (fixed zero)',
            physics=dict(engine='MuJoCo', version=mujoco.__version__, revision=3 if self.fixed_lift is not None else 2, timestep=.002,
                fixed_lift_m=self.fixed_lift, lift_dof_removed=self.fixed_lift is not None,
                solver=dict(multiccd=True, cone='elliptic', impratio=10, noslip_iterations=5, iterations=60),
                collision='robot-object, robot-environment, object-object; no robot self collision',
                object_mass_kg=[.12,.12,.12,.10], friction=[.8,.01,.001], condim=6,
                banana_collision='single convex hull', actuator_parameters='approximate, not calibrated'),
            camera_mount_status='provisional simulated mounts; real hardware calibration required',
            excluded_operator_cameras=['overview'],
            camera_calibration={name: dict(parent=self.model.body(int(self.model.cam_bodyid[self.model.camera(name).id])).name,
                local_pos=self.model.cam_pos[self.model.camera(name).id].tolist(),
                local_quat_wxyz=self.model.cam_quat[self.model.camera(name).id].tolist(),
                fovy=float(self.model.cam_fovy[self.model.camera(name).id])) for name in CAMERAS})

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
