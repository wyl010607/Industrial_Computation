import time
import os


class TimeRecorder:
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
