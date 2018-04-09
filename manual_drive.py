import gym
import numpy as np
import imageio
import os
from pyglet.window import key

env = gym.make('CarRacing-v0').env
env.reset()

if __name__=='__main__':
    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0

    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    env.reset()
    while True:
        s, r, done, info = env.step(a)
        env.render()
        if done:
            env.reset()
    env.close()
