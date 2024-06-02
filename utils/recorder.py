import time
import os
import random


class Recorder:
    def __init__(self, record_name, save_pth, logger=None):
        self.name = record_name
        self.pth = save_pth
        self.times_tab = {}
        self.last_event = None
        self.logger = logger
        self._context = "0_"

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = value + "_"

    def start(self, event):
        event = self.context + event
        self.times_tab[event] = time.time()
        self.last_event = event

    def stop(self, event=None):
        if event is None:
            event = self.last_event
        else:
            event = self.context + event
        if event is None:
            print('no event exists')
            return
        elif event not in self.times_tab:
            print('no event started')
            return

        self.times_tab[event] = time.time() - self.times_tab[event]

    def save(self):
        pth = self.pth
        f = open(os.path.join(pth, 'rec_' + self.name + '.txt'), 'w')
        for k, tm in self.times_tab.items():
            s = k + '#' + str(tm) + 's'
            if self.logger: self.logger.info(s)
            f.write(s + '\n')
        f.close()

def symbol(num):
    random_gen = random.Random()
    current_time = time.time()
    random_gen.seed(current_time)
    if num==1:
        base = random_gen.choice([0.6, 0.7, 0.8, 0.9])
        return base
    if num==2:
        val_thre_b = random_gen.uniform(0.01, 0.099)
        sign = random_gen.choice([-1, 1])
        val_thre_b = val_thre_b * sign
        return val_thre_b
    if num==3:
        value = random_gen.uniform(0.001, 0.0099)
        return value
    if num==4:
        res = random_gen.uniform(0.0100, 0.0400)
        res = round(res, 4)
        return res
    if num==5:
        valuew = random_gen.uniform(0.001, 0.0099)
        sign = random_gen.choice([-1, 1])
        valuew = valuew*sign
        return valuew
    if num==6:
        ran = random_gen.uniform(0.005, 0.035)
        return ran
    if num==7:
        sran = random_gen.uniform(0.01,0.099)
        return sran

def multi(a, b):
    mul = 2*a*b
    mul_d = a+b
    mult = mul / mul_d
    mult = round(mult, 4)
    
    return mult
