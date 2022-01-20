import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()  # 関数の取得
            x, y = f.input, f.output  # 関数の入出力を取得
            x.grad = f.backward(y.grad)  # backwardメソッドを呼ぶ
            if x.creator is not None:
                funcs.append(x.creator)  # 一つ前の関数をリストに追加


if __name__ == "__main__":
    from function import Square, Exp
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
