import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

if __name__ == "__main__":
    from core import Variable, as_array

    class Function:
        def __call__(self, inputs) -> int:
            xs = [x.data for x in inputs]
            ys = self.forward(xs)
            outputs = [Variable(as_array(y)) for y in ys]
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = outputs
            return outputs

        def forward(self, xs):
            raise NotImplementedError()

        def backward(self, gys):
            raise NotImplementedError()

    class Add(Function):
        def forward(self, xs):
            x0, x1 = xs
            y = x0 + x1
            return (y,)
    xs = [Variable(np.array(2)), Variable(np.array(3))]
    f = Add()
    ys = f(xs)
    y = ys[0]
    print(y.data)
