"""Gamepad and keyboard inputs; no robot communication."""
import ctypes as c
import json
import os
import time
from pathlib import Path

import numpy as np


class Disconnected(RuntimeError):
    pass


class Gamepad:
    def __init__(self, config):
        self.config = config
        self.slot = config.get('slot', 0)
        self.previous = 0
        self.armed = False
        if os.name == 'nt':
            from .windows_gamepad import Caps, Joy
            self.Joy = Joy
            self.mm = c.WinDLL('winmm')
            self.caps = Caps()
            if self.mm.joyGetDevCapsW(self.slot, c.byref(self.caps), c.sizeof(self.caps)) or self.caps.axes < 2:
                raise Disconnected('No readable gamepad axes. Run outside sandbox or use --input keyboard.')
            self.name = self.caps.name
            self.backend = 'winmm'
        else:
            # WSL USB forwarding is optional. Keyboard remains available without it.
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() <= self.slot:
                raise Disconnected('No SDL gamepad. WSL may need USB forwarding; use --input keyboard.')
            self.pad = pygame.joystick.Joystick(self.slot)
            self.pad.init()
            self.pygame = pygame
            self.name, self.backend = self.pad.get_name(), 'sdl'

    def raw(self):
        if self.backend == 'winmm':
            pos = self.Joy(size=c.sizeof(self.Joy), flags=0xff)
            if self.mm.joyGetPosEx(self.slot, c.byref(pos)):
                self.armed = False
                raise Disconnected('Gamepad disconnected')
            axes = {}
            for name in 'xyzruv':
                low, high = getattr(self.caps, name+'min'), getattr(self.caps, name+'max')
                axes[name] = 2*(getattr(pos, name)-low)/max(1, high-low)-1
            return axes, int(pos.buttons), int(pos.pov)
        self.pygame.event.pump()
        if not self.pad.get_attached():
            raise Disconnected('Gamepad disconnected')
        axes = {str(i): self.pad.get_axis(i) for i in range(self.pad.get_numaxes())}
        buttons = sum(self.pad.get_button(i) << i for i in range(self.pad.get_numbuttons()))
        hat = self.pad.get_hat(0) if self.pad.get_numhats() else (0, 0)
        pov = {(0, 1): 0, (1, 0): 9000, (0, -1): 18000, (-1, 0): 27000}.get(hat, 65535)
        return axes, buttons, pov

    def read(self):
        axes, buttons, pov = self.raw()
        def pressed(name):
            index = self.config['buttons'][name] - 1
            return bool(buttons & (1 << index))
        def edge(name):
            index = self.config['buttons'][name] - 1
            return pressed(name) and not (self.previous & (1 << index))
        deadman = pressed('deadman')
        if not deadman:
            self.armed = True  # Must see release once after connect.
        enabled = self.armed and deadman
        mapping = self.config['axes' if self.backend == 'winmm' else 'sdl_axes']
        def axis(name):
            key, sign = mapping[name]
            value = axes.get(str(key), 0) * sign
            dz = self.config['deadzone']
            return np.sign(value)*max(0, abs(value)-dz)/(1-dz)
        motion = np.array([axis('forward'), axis('left'), axis('up')])*.12
        rotation = np.array([0, 0, axis('yaw')])*.5
        if pressed('rotate_mode'):
            rotation = np.array([axis('left'), axis('forward'), axis('yaw')])*.5
            motion[:] = 0
        result = dict(translation=motion if enabled else np.zeros(3),
            rotation=rotation if enabled else np.zeros(3),
            gripper=.05*(int(pressed('open'))-int(pressed('close'))) if enabled else 0,
            lift=(.05 if pov == 0 else -.05 if pov == 18000 else 0) if enabled else 0,
            events=[name for name in ('switch_arm', 'record', 'save', 'discard') if edge(name)],
            enabled=enabled, mode='ROTATE' if pressed('rotate_mode') else 'MOVE')
        self.previous = buttons
        return result


class Keyboard:
    def __init__(self):
        from pynput import keyboard
        self.held, self.edges = set(), set()
        def name(key):
            try:
                return key.char.lower()
            except (AttributeError, TypeError):
                return key.name
        def press(key):
            k = name(key)
            if k not in self.held:
                self.edges.add(k)
            self.held.add(k)
        def release(key):
            self.held.discard(name(key))
        self.listener = keyboard.Listener(on_press=press, on_release=release)
        self.listener.start()

    def read(self):
        held = self.held.copy()
        edges = self.edges.copy()
        self.edges.difference_update(edges)
        enabled = 'space' in held
        def pair(a, b):
            return int(a in held)-int(b in held)
        return dict(translation=np.array([pair('w','s'), pair('a','d'), pair('r','f')])*.12 if enabled else np.zeros(3),
            rotation=np.array([pair('u','o'), pair('i','k'), pair('q','e')])*.5 if enabled else np.zeros(3),
            gripper=.05*pair('z','x') if enabled else 0,
            lift=.05*pair('page_up','page_down') if enabled else 0,
            enabled=enabled, mode='KEYBOARD',
            events=[event for key, event in [('tab','switch_arm'),('f5','record'),('f6','save'),
                ('f7','discard'),('f8','reset'),('f9','keyboard'),('esc','quit')] if key in edges])

    def close(self):
        self.listener.stop()


def load_config(path=None):
    path = Path(path) if path else Path(__file__).with_name('gamepad_config.json')
    return json.loads(path.read_text(encoding='utf-8'))


def monitor(config, seconds):
    pad = Gamepad(config)
    print(f'{pad.backend}: {pad.name}', flush=True)
    deadline = time.monotonic() + seconds
    previous = None
    while time.monotonic() < deadline:
        axes, buttons, pov = pad.raw()
        value = ({k: round(v, 2) for k, v in axes.items()}, buttons, pov)
        if value != previous:
            print(value, flush=True)
            previous = value
        time.sleep(.05)
