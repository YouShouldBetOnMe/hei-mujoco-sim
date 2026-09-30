"""Run with python -m hei_sim.collect. Simulation only."""
import argparse
from pathlib import Path
import time

import cv2
import mujoco.viewer
import numpy as np

from .controls import Disconnected, Gamepad, Keyboard, load_config, monitor
from .environment import CAMERAS, Environment
from .recording import EpisodeWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', choices=['auto','gamepad','keyboard'], default='auto')
    parser.add_argument('--output', type=Path, default=Path('datasets/hei_raw'))
    parser.add_argument('--task', default='Move the red cube to the center of the table')
    parser.add_argument('--gamepad-config')
    parser.add_argument('--monitor-gamepad', type=float, default=0, metavar='SECONDS')
    parser.add_argument('--smoke-frames', type=int, default=0, help='Headless test data, never a task demonstration')
    parser.add_argument('--preview-seconds', type=float, default=0, help='Close interactive preview after this duration')
    args = parser.parse_args()
    config = load_config(args.gamepad_config)
    if args.monitor_gamepad:
        monitor(config, args.monitor_gamepad)
        return
    env = Environment()
    writer = None
    keyboard = None
    try:
        if args.smoke_frames:
            writer = EpisodeWriter(args.output, env.metadata(), 'TEST ONLY: stationary simulation', test=True)
            for _ in range(args.smoke_frames):
                observation = env.observe()
                action = env.target.copy()
                env.step(action)
                writer.add(observation, action)
            print('Test episode:', writer.save(), flush=True)
            return
        keyboard = Keyboard()
        pad = None
        if args.input != 'keyboard':
            try:
                pad = Gamepad(config)
                print(f'Gamepad: {pad.name} ({pad.backend})', flush=True)
            except (Disconnected, ImportError) as exc:
                if args.input == 'gamepad':
                    raise
                print(f'{exc}\nUsing keyboard.', flush=True)
        mode = 'gamepad' if pad else 'keyboard'
        side = 'right'
        print('F5 record / F6 save / F7 discard / F8 reset / F9 keyboard / TAB arm / ESC quit', flush=True)
        print('Keyboard: hold SPACE; WASD XY; R/F Z; Q/E yaw; U/O roll; I/K pitch; Z/X open/close.', flush=True)
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.lookat[:] = [.45, 0, .8]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.7, 135, -25
            running = True
            preview_started = time.monotonic()
            while running and viewer.is_running():
                if args.preview_seconds and time.monotonic()-preview_started >= args.preview_seconds:
                    break
                start = time.monotonic()
                keys = keyboard.read()
                events = list(keys['events'])
                control = keys
                if mode == 'gamepad':
                    try:
                        control = pad.read()
                        events.extend(control['events'])
                    except Disconnected as exc:
                        print(str(exc) + '; recording discarded, keyboard enabled.', flush=True)
                        if writer:
                            writer.discard()
                            writer = None
                        mode = 'keyboard'
                        # Hold the last target; never apply stale gamepad data.
                        control = dict(translation=np.zeros(3), rotation=np.zeros(3), gripper=0, lift=0)
                for event in set(events):
                    if event == 'quit': running = False
                    elif event == 'switch_arm': side = 'left' if side == 'right' else 'right'
                    elif event == 'record' and writer is None:
                        writer = EpisodeWriter(args.output, {**env.metadata(), 'controller':mode,
                            'controller_mapping':config if mode == 'gamepad' else 'documented keyboard mapping'},
                            args.task, test=bool(args.preview_seconds))
                        print('Recording started', flush=True)
                    elif event == 'save' and writer:
                        if writer.count:
                            print('Saved:', writer.save(), flush=True)
                        else:
                            writer.discard()
                        writer = None
                    elif event == 'discard' and writer:
                        writer.discard()
                        writer = None
                    elif event == 'reset' and writer is None: env.reset()
                    elif event == 'keyboard':
                        if writer:
                            writer.discard()
                            writer = None
                        mode = 'keyboard'
                        control = keys
                if not running: break
                with viewer.lock():
                    observation = env.observe()
                    action = env.command(side, control['translation'], control['rotation'], control['gripper'], control['lift'])
                    env.step(action)
                if writer:
                    writer.add(observation, action)
                viewer.sync()
                panels = []
                for name in CAMERAS:
                    panel = cv2.cvtColor(observation['images'][name], cv2.COLOR_RGB2BGR)
                    cv2.putText(panel, name, (8,20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                    panels.append(panel)
                strip = np.hstack(panels)
                # Draw controls outside the RGB images; this HUD is not recorded.
                hud = np.zeros((96, strip.shape[1], 3), dtype=np.uint8)
                enabled = control.get('enabled', False)
                control_mode = control.get('mode', 'KEYBOARD')
                status = f'REC {writer.count}' if writer else 'IDLE'
                grip_index = 6 if side == 'right' else 13
                lift_limit = ' | LIFT AT TOP' if env.target[17] >= -1e-5 else ''
                lines = [f'{mode.upper()} {side.upper()} | {control_mode} | {"ENABLED" if enabled else "HOLD LB / SPACE"} | {status}',
                    f'Grip actual/target: {env.state()[grip_index]*1000:.0f}/{env.target[grip_index]*1000:.0f} mm{lift_limit}']
                if mode == 'gamepad':
                    lines += ['LB + right stick: hand up/down | LB + D-pad: lift',
                              'LB+RB + left stick forward/back: wrist pitch | F5 REC F6 SAVE']
                else:
                    lines += ['SPACE + R/F: hand up/down | I/K: wrist pitch | PgUp/PgDn: lift',
                              'Z/X: open/close | TAB: arm | F5 REC F6 SAVE F7 DISCARD']
                for i, line in enumerate(lines):
                    cv2.putText(hud, line, (8,19+i*23), cv2.FONT_HERSHEY_SIMPLEX, .44, (0,255,255), 1)
                cv2.imshow('HEI 3 cameras | F5 record F6 save F7 discard', np.vstack([strip,hud]))
                cv2.waitKey(1)
                time.sleep(max(0, 1/env.fps-(time.monotonic()-start)))
    finally:
        if writer and not writer.closed:
            writer.discard()
        if keyboard:
            keyboard.close()
        env.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
