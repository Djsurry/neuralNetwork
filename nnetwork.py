import time
import random, math
import sys
from PIL import Image
import pygame
import time
import numpy as np
from random_words import RandomWords
import json
import enchant

clock = pygame.time.Clock()
pygame.init()

WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,155, 0)
YELLOW = (255, 215, 0)
BRIGHTYELLOW = (255, 255, 0)
BLUE = (0,0, 255)
BRIGHTGREEN = (0, 255 , 0)
DARKRED = (200, 0 , 0 )

fps = 30
font1 = pygame.font.SysFont(None, 50)
font2 = pygame.font.SysFont(None, 25)

displayWidth = 800
displayHieght = 600

# -----------JSON WEIGHT FORMAT--------------

#   [(id_of_input, val, id_of_output), ...]

# -------------------------------------------
gameDisplay = pygame.display.set_mode((displayWidth,displayHieght))

pygame.display.set_caption('Nueral Network')

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def pos(x):
    if x <= 0:
        return 0
    else:
        return x

def sigmoid(x):
    return np.tanh(x)


def pickColor(x):
    return (0,0,0)
class Weight:
    def __init__(self, input_, val, output):
        self.input = input_
        self.val = val 
        self.output = output

    def change(self):
        return self.input.val*self.val


class Node:
    def __init__(self, inputs, id_, n):
        self.inputs = inputs
        self.val = 0
        self.color = pickColor(self.val)
        self.id = id_
        self.network = n
        self.balanceType = None
    def changeType(self, t):
        self.balanceType = t

class Input(Node):
    def __init__(self, id_, n):

        Node.__init__(self, None, id_, n)
        self.val = None

    def run(self, val):
        self.val = val
        self.color = pickColor(self.val)


class Hidden(Node):
    def __init__(self, inputs, id_, n):
        Node.__init__(self, inputs, id_, n)

    def run(self):
        vals = []
        for i in self.inputs:
            
            for j in self.network.weights:
                
                if j.input == i and j.output == self:
                   
                    weight = j
                    
            vals.append(weight.change())
        val = sum(vals)
        for i in self.balanceType:
            if i == 'sigmoid':
                val = sigmoid(val)
            elif i == 'pos':
                val = pos(val)
            elif i == None:
                val = val
        self.val = val
        self.color = pickColor(self.val)


class Output(Node):
    def __init__(self, inputs, meaning, n):
        Node.__init__(self, inputs, None, n)
        self.meaning = meaning
        self.confidence = None

    def run(self):
        vals = []
        for i in self.inputs:
            for j in self.network.weights:
                if j.input == i and j.output == self:
                    weight = j
                    break
            vals.append(weight.change())
        if self.balanceType == 'sigmoid':
            self.val = sigmoid(sum(vals))
        elif self.balanceType == 'center':
            self.val = center(sum(vals))
        elif self.balanceType == None:
            self.val = sum(vals)
        self.color = pickColor(self.val)


class Network:
    def __init__(self):
        self.ids = {}
        self.weights = [
        ]
        self.inputLayer = [Input(i, self) for i in range(1, 5)]
        for i in self.inputLayer:
            self.ids[i.id] = i
        self.h1 = [
            Hidden([self.ids[1], self.ids[2]], 5, self), 
            Hidden([self.ids[2], self.ids[3]], 6, self), 
            Hidden([self.ids[1], self.ids[3]], 7, self),
            Hidden([self.ids[2], self.ids[4]], 8, self),
            Hidden([self.ids[1], self.ids[4]], 9, self),
            Hidden([self.ids[3], self.ids[4]], 10, self)
        ]
        for i in self.h1:
            self.ids[i.id] = i
        self.h2 = [
            Hidden([self.ids[5], self.ids[10]], 11, self),
            Hidden([self.ids[5], self.ids[10]], 12, self),
            Hidden([self.ids[7], self.ids[8]], 13, self),
            Hidden([self.ids[7], self.ids[8]], 14, self),
            Hidden([self.ids[6], self.ids[9]], 15, self),
            Hidden([self.ids[6], self.ids[9]], 16, self),
            Hidden([self.ids[5], self.ids[10]], 17, self),
            Hidden([self.ids[5], self.ids[10]], 18, self)
        ]
        for i in self.h2:
            self.ids[i.id] = i
        self.outputLayer = [
            Output([self.ids[11]], 'Empty', self),
            Output([self.ids[13], self.ids[14]], 'Diagonal', self),
            Output([self.ids[15], self.ids[16]], 'Vertical', self),
            Output([self.ids[17], self.ids[12]], 'Horizontal', self),
            Output([self.ids[18]], 'Full', self)
        ]

        self.layers = [self.inputLayer, self.h1, self.h2, self.outputLayer]

        # for i in self.layers:
        #     if i != self.inputLayer:
        #         for j in i:
        #             for link in j.inputs:
        #                 self.weights.append(Weight(link[0], link[1], j))

        self.weightConstructor()
        for i in self.h1:
            i.changeType('sigmoid')
        for i in self.h2:
            i.changeType(['sigmoid', 'pos'])
        
        


    def weightConstructor(self):
        l = json.load(open('weights.json'))
        self.weights = []
        for i, v, o in l:
            try:
                self.weights.append(Weight(self.ids[i], v, self.ids[o]))
            except KeyError:
                for j in self.outputLayer:
                    if j.meaning == o:
                        
                        self.weights.append(Weight(self.ids[i], v, j))
                        break
    def dumpWeights(self):
        dump = []
        for i in self.weights:
            if i.output.id == None:
                dump.append((i.input.id, i.val, i.output.meaning))
            else:
                dump.append((i.input.id, i.val, i.output.id))
        json.dump(dump, open('weights.json', 'w'))
    
    def run(self):
        self.img = Image.open('test.png').convert('L')
        WIDTH, HEIGHT = self.img.size
        data = list(self.img.getdata())
        data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]
       
        self.inputLayer[0].run(translate(data[0][0], 0, 255, -1, 1))
        self.inputLayer[1].run(translate(data[0][1], 0, 255, -1, 1))
        self.inputLayer[2].run(translate(data[1][1], 0, 255, -1, 1))
        self.inputLayer[3].run(translate(data[1][0], 0, 255, -1, 1))
        for i in self.inputLayer:
            print(str(i.id) + ': ' +str(i.val))
        print()
        for i in self.h1:
            i.run()
        for i in self.h1:
            print(str(i.id) + ': ' +str(i.val))
        print()
        for i in self.h2:
            i.run()
        for i in self.h2:
            print(str(i.id) + ': ' +str(i.val))
        print()
        for i in self.outputLayer:
            i.run()

        for i in self.outputLayer:
            print(i.meaning + ': ' + str(i.val))

        

n = Network()
n.run()


