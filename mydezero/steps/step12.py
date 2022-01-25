import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

if __name__ == "__main__":
    from core import Variable, as_array

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
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = outputs
            return outputs if len(outputs) > 1 else outputs[0]

        def forward(self, xs):
            raise NotImplementedError()

        def backward(self, gys):
            raise NotImplementedError()

    class Add(Function):
        def forward(self, x0, x1):
            y = x0 + x1
            return y

    def add(x0, x1):
        return Add()(x0, x1)

    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    f = Add()
    y = f(x0, x1)
    print(y.data)
    y = add(x0, x1)
    print(y.data)
