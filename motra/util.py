from cmath import sqrt


def distance(x0, y0, x1, y1):
    return sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2))
