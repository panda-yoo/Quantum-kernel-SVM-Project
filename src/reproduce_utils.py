import numpy as np

def generate_xor(N=100, std=0.25, seed=42):
    np.random.seed(seed)
    
    # 4 centers
    centers = np.array([
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1]
    ])
    
    X = []
    y = []
    
    samples_per_cluster = N // 4
    
    for cx, cy in centers:
        # sample points
        points = np.random.normal(loc=[cx, cy], scale=std, size=(samples_per_cluster, 2))
        
        X.append(points)
        
        # XOR label
        labels = np.sign(points[:, 0] * points[:, 1])
        y.append(labels)
    
    X = np.vstack(X)
    y = np.hstack(y)
    
    return X, y