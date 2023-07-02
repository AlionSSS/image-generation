import time
import numpy as np
import torchvision as tv
import visdom


class Visualizer():
    """
    封装Visdom基本操作
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot(self, name, y, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=name, opts=dict(title=name),
                      update=None if x == 0 else "append", **kwargs)
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.images(img_.cpu().numpy(), win=name, opts=dict(title=name), **kwargs)

    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log_text'):
        self.log_text += f"[{time.strftime('%m%d_%H%M%S')}] {info} <br>"
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)
