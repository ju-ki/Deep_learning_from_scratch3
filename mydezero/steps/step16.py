import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core import as_array
from function import square, add


class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        # funcs = [self.creator]
        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
        add_func(self.creator)

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
                    add_func(x.creator)


class Function(object):
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
        self.generation = max([x.generation for x in inputs])

        for output in outputs:
            output.set_creator(self)  # 出力変数に生みの親を覚えさせる
        self.inputs = inputs
        self.outputs = outputs  # 出力も覚える
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


if __name__ == "__main__":
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()
    print(y.data)
    print(x.grad)