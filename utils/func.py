import numpy as np

def sample_ch(meta_batch):
    judge = 0
    if meta_batch>=50:
        judge = 0
    else:
        judge = 1
        
    return judge