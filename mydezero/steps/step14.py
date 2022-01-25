import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from function import add


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            # 関数の入出力を取得(リストにまとめる)
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            # 順伝播の改善と同様(addクラスの逆伝播に対応)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                # x.grad = gx  # ここが違う
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
                if x.creator is not None:
                    funcs.append(x.creator)

if __name__ == "__main__":
    x = Variable(np.array(3.0))
    y = add(x, x)
    print("y", y.data)
    y.backward()
    print("x.grad", x.grad)
    # x = Variable(np.array(3.0))
    x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    print(x.grad)