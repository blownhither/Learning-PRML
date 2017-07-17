import numpy as np
from scipy.spatial import KDTree

def test_KDTree():
    data = np.random.random((2000, 5))
    k = KDTree(data)
    print(k.query(data[0] + 1e-10))
    ans = k.query_ball_point(data[0], 0.2)
    print(ans)



if __name__ == '__main__':
    test_KDTree()

