"""Read-only Windows joystick inventory (no simulator or robot connection)."""
import ctypes as c
import json
from ctypes import wintypes as w


class Caps(c.Structure):
    _fields_ = [('mid', w.WORD), ('pid', w.WORD), ('name', w.WCHAR * 32)] + [
        (n, w.UINT) for n in ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax',
        'buttons', 'period_min', 'period_max', 'rmin', 'rmax', 'umin', 'umax',
        'vmin', 'vmax', 'caps', 'max_axes', 'axes', 'max_buttons')
    ] + [('regkey', w.WCHAR * 32), ('oem', w.WCHAR * 260)]


class Joy(c.Structure):
    _fields_ = [(n, w.DWORD) for n in ('size', 'flags', 'x', 'y', 'z', 'r', 'u', 'v',
                                     'buttons', 'button_number', 'pov', 'reserved1', 'reserved2')]


def inventory():
    mm = c.WinDLL('winmm')
    found = []
    for slot in range(mm.joyGetNumDevs()):
        pos = Joy(size=c.sizeof(Joy), flags=0xff)
        if mm.joyGetPosEx(slot, c.byref(pos)) == 0:
            cap = Caps()
            if mm.joyGetDevCapsW(slot, c.byref(cap), c.sizeof(cap)) != 0:
                raise RuntimeError('Cannot read joystick capabilities')
            found.append(dict(slot=slot, name=cap.name, axes=cap.axes, buttons=cap.buttons,
                ranges={a: [getattr(cap, a+'min'), getattr(cap, a+'max')] for a in 'xyzruv'},
                values={a: getattr(pos, a) for a in 'xyzruv'}, pressed=pos.buttons, pov=pos.pov))
    return found


if __name__ == '__main__':
    print(json.dumps(inventory(), ensure_ascii=False, indent=2))
