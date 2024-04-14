import numpy
import numpy as np
from enum import IntEnum
import torch, os 

class discretization:
    def __init__(self, naction=3, aceel_div=100):
        """
        naction: Number of actions
        K = Number of bins to create 
        """
        self.naction = naction
        # action_space: {Acceleration, Steering, Brake)
        self.K = [aceel_div, 3, 2]
        self.K_n = [aceel_div+1, 3, 2]
        self.start = [0, 0, 0]
        self.action_low = [0, -1, 0]
        self.action_high = [1, 1, 1]
        self.accel_div = aceel_div
        self.bins = [np.round(np.linspace(self.action_low[i], self.action_high[i], self.K[i], dtype=float), 3) for i in range(self.naction)]

    def __call__(self, x):
        # https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
        # inds =
        # TODO remove tensor to np conversion
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        closest_actions_idx = [np.abs(self.bins[i] - x[i]).argmin() for i in range(self.naction)]
        # discretized_value = [self.bins[i][inds[i]-1] for i in range(self.naction)]
        # return self.bins[closest_actions]
        res = torch.tensor([self.bins[i][j] for i, j in enumerate(closest_actions_idx)])
        res[0] *= self.accel_div
        res[0] = int(res[0])
        if res[1] == -1.0:
            res[1] = 1.0
        elif res[1] == 1.0:
            res[1] = 2.0
        res[1] = int(res[1])
        res[2] = int(res[2])
        return res

    def de_discrete(self, action: np.ndarray) -> np.ndarray:
        res = []

        res.append(float(action[0])/float(self.accel_div))
        if action[1] == 2.0:
            action[1] = 1.0
        elif action[1] == 1.0:
            action[1] = -1.0
        res.append(float(action[1]))
        res.append(float(action[2]))
        return np.array(res)

def load_policy(dagger_trainer, path, ckpt='hockey'):
    ckptPath = f"{path}/{ckpt}.pt"
    if os.path.exists(ckptPath):
        # print(f"Updated the states using: {ckptPath}")
        checkpoint = torch.load(ckptPath)
        dagger_trainer.policy.load_state_dict(checkpoint['state_dict']) 
    return dagger_trainer

class Team(IntEnum):
    RED = 0
    BLUE = 1


def video_grid(team1_images, team2_images, team1_state='', team2_state=''):
    from PIL import Image, ImageDraw
    grid = np.hstack((np.vstack(team1_images), np.vstack(team2_images)))
    grid = Image.fromarray(grid)
    grid = grid.resize((grid.width // 2, grid.height // 2))

    draw = ImageDraw.Draw(grid)
    draw.text((20, 20), team1_state, fill=(255, 0, 0))
    draw.text((20, grid.height // 2 + 20), team2_state, fill=(0, 0, 255))
    return grid


def map_image(team1_state, team2_state, soccer_state, resolution=512, extent=65, anti_alias=1):
    BG_COLOR = (0xee, 0xee, 0xec)
    RED_COLOR = (0xa4, 0x00, 0x00)
    BLUE_COLOR = (0x20, 0x4a, 0x87)
    BALL_COLOR = (0x2e, 0x34, 0x36)
    from PIL import Image, ImageDraw
    r = Image.new('RGB', (resolution*anti_alias, resolution*anti_alias), BG_COLOR)

    def _to_coord(x):
        return resolution * anti_alias * (x + extent) / (2 * extent)

    draw = ImageDraw.Draw(r)
    # Let's draw the goal line
    draw.line([(_to_coord(x), _to_coord(y)) for x, _, y in soccer_state['goal_line'][0]], width=5*anti_alias, fill=RED_COLOR)
    draw.line([(_to_coord(x), _to_coord(y)) for x, _, y in soccer_state['goal_line'][1]], width=5*anti_alias, fill=BLUE_COLOR)

    # and the ball
    x, _, y = soccer_state['ball']['location']
    s = soccer_state['ball']['size']
    draw.ellipse((_to_coord(x-s), _to_coord(y-s), _to_coord(x+s), _to_coord(y+s)), width=2*anti_alias, fill=BALL_COLOR)

    # and karts
    for c, s in [(BLUE_COLOR, team1_state), (RED_COLOR, team2_state)]:
        for k in s:
            x, _, y = k['kart']['location']
            fx, _, fy = k['kart']['front']
            sx, _, sy = k['kart']['size']
            s = (sx+sy) / 2
            draw.ellipse((_to_coord(x - s), _to_coord(y - s), _to_coord(x + s), _to_coord(y + s)), width=5*anti_alias, fill=c)
            draw.line((_to_coord(x), _to_coord(y), _to_coord(x+(fx-x)*2), _to_coord(y+(fy-y)*2)), width=4*anti_alias, fill=0)

    if anti_alias == 1:
        return r
    return r.resize((resolution, resolution), resample=Image.ANTIALIAS)


# Recording functionality
class BaseRecorder:
    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        raise NotImplementedError

    def __and__(self, other):
        return MultiRecorder(self, other)

    def __rand__(self, other):
        return MultiRecorder(self, other)


class MultiRecorder(BaseRecorder):
    def __init__(self, *recorders):
        self._r = [r for r in recorders if r]

    def __call__(self, *args, **kwargs):
        for r in self._r:
            r(*args, **kwargs)


class VideoRecorder(BaseRecorder):
    """
        Produces pretty output videos
    """
    def __init__(self, video_file):
        import imageio
        self._writer = imageio.get_writer(video_file, fps=20)

    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        if team1_images and team2_images:
            self._writer.append_data(np.array(video_grid(team1_images, team2_images,
                                                         'Blue: %d' % soccer_state['score'][1],
                                                         'Red: %d' % soccer_state['score'][0])))
        else:
            self._writer.append_data(np.array(map_image(team1_state, team2_state, soccer_state)))

    def __del__(self):
        if hasattr(self, '_writer'):
            self._writer.close()


class DataRecorder(BaseRecorder):
    def __init__(self, record_images=False):
        self._record_images = record_images
        self._data = []

    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        data = dict(team1_state=team1_state, team2_state=team2_state, soccer_state=soccer_state, actions=actions)
        if self._record_images:
            data['team1_images'] = team1_images
            data['team2_images'] = team2_images
        self._data.append(data)

    def data(self):
        return self._data

    def reset(self):
        self._data = []


class StateRecorder(BaseRecorder):
    def __init__(self, state_action_file, record_images=False):
        self._record_images = record_images
        self._f = open(state_action_file, 'wb')

    def __call__(self, team1_state, team2_state, soccer_state, actions, team1_images=None, team2_images=None):
        from pickle import dump
        data = dict(team1_state=team1_state, team2_state=team2_state, soccer_state=soccer_state, actions=actions)
        if self._record_images:
            data['team1_images'] = team1_images
            data['team2_images'] = team2_images
        dump(dict(data), self._f)
        self._f.flush()

    def __del__(self):
        if hasattr(self, '_f'):
            self._f.close()


def load_recording(recording):
    from pickle import load
    with open(recording, 'rb') as f:
        while True:
            try:
                yield load(f)
            except EOFError:
                break

