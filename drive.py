import gym
import numpy as np
import sys
import torch
from torch.autograd import Variable
import PIL
from torch.nn import Softmax
from pyglet.window import key

from model import CustomModel
from data import transform_driving_image, LEFT, RIGHT, GO

id_to_steer = {
    LEFT: -1,
    RIGHT: 1,
    GO: 0,
}

if __name__=='__main__':

    if len(sys.argv) < 2:
        sys.exit("Usage : python drive.py path/to/weights")
    # load the model
    model_weights = sys.argv[1]
    model = CustomModel()
    model.load_state_dict(torch.load(model_weights))

    env = gym.make('CarRacing-v0').env
    env.reset()

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        global restart
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
    
    # initialisation
    for i in range(50):
        env.step([0, 0, 0])
        env.render()
    
    i = 0
    while True:
        s, r, done, info = env.step(a)
        s = s.copy()
        # We transform our numpy array to PIL image
        # because our transformation takes an image as input
        s  = PIL.Image.fromarray(s)  
        input = transform_driving_image(s)
        input = Variable(input[None, :], volatile=True)
        output = Softmax()(model(input))
        _, index = output.max(1)
        index = index.data[0]
        a[0] = id_to_steer[index] * output.data[0, index] * 0.3  # lateral acceleration
        env.render()
    env.close()

    
