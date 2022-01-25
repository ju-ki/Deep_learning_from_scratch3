import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def square(x):
    f = Square()
    return f(x)


def exp(x):
    f = Exp()
    return f(x)


if __name__ == "__main__":
    from function import Square, Exp
    from core import Variable
    x = Variable(np.array(0.5))
    a = square(x)
    b = exp(a)
    y = square(b)
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
    # backwardメソッドの簡略化の確認
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad)
    # isinstanceの確認
    x = Variable(1.0)
