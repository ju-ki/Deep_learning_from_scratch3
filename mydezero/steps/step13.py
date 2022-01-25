import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

if __name__ == "__main__":
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
                    x.grad = gx
                    if x.creator is not None:
                        funcs.append(x.creator)

    def as_array(x):
        if np.isscalar(x):
            return np.array(x)
        return x

    class Function:
        def __call__(self, *inputs) -> int:
            # アスタリスクをつけることで可変長に対応する
            xs = [x.data for x in inputs]
            ys = self.forward(*xs)  # リストの要素を展開する
            """"
                xs = [x0, x1]
                self.forward(*xs) --> self.forward(x0, x1)
            """
            if not isinstance(ys, tuple):  # 返す要素が一つの時タプルに変更してそのまま渡す
                ys = (ys,)
            outputs = [Variable(as_array(y)) for y in ys]
            for output in outputs:
                output.set_creator(self)  # 出力変数に生みの親を覚えさせる
            self.inputs = inputs
            self.outputs = outputs  # 出力も覚える
            return outputs if len(outputs) > 1 else outputs[0]

        def forward(self, xs):
            raise NotImplementedError()

        def backward(self, gys):
            raise NotImplementedError()

    class Square(Function):
        def forward(self, x):
            y = x ** 2
            return y

        def backward(self, gy):
            x = self.inputs[0].data
            gx = 2 * x * gy
            return gx

    class Add(Function):
        def forward(self, x0, x1):
            y = x0 + x1
            return y

        def backward(self, gy):
            return gy, gy

    def square(x):
        return Square()(x)

    def add(x0, x1):
        return Add()(x0, x1)

    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))

    z = add(square(x), square(y))
    z.backward()
    print(z.data)
    print(x.grad)
    print(y.grad)
