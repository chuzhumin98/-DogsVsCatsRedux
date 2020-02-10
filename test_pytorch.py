import torch
import numpy as np

if __name__ == '__main__':
    a = torch.from_numpy(np.array(range(10)))
    if torch.cuda.is_available():
        a = a.cuda()


    print(a)