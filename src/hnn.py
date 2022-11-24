import numpy as np
import matplotlib.pyplot as plt
np.random.seed(5)


""" 重み行列Wを計算 """
def fit(dim, data):
    w = np.zeros([dim, dim])
 
    for i in data:
        w += np.outer(i, i.T)
    for i in range(dim):
        w[i][i] = 0     # 対角成分を0にする
    return w

""" 学習データにノイズを与え、テストデータを作成 """
def noise(data, rate):
    test_data = np.copy(data)
    inv = np.random.binomial(n=1, p=rate, size=len(data))
    for i, v in enumerate(data):
        if inv[i]:
            test_data[i] = -1 * v
    return test_data

''' エネルギー関数E(x)を定義し、収束するまで繰り返す '''
def energy(data, w):
    e = -0.5 * np.dot(w, np.dot(data, data.T))

""" エネルギーが変化しなくなるまで更新を行う """
def predict(data, w, loop=100):
    e = energy(data, w)
    for i in range(loop):
        data = np.dot(w, data)
        # xr の符号をとる
        data = np.sign(data)
        e_new = energy(data, w)
        if e == e_new:
            return data
        e = e_new
        if loop == 50:
            flaten_e = e.reshape(9, 7)
            flaten_e.savefig('img.png')
            plt.imshow(flaten_e)
            plt.gray()
            plt.show()
        
    return data


def main():
    two = np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [0,+1,+1,+1,+1,+1, 0],
                [0, 0, 0, 0, 0,+1, 0],
                [0, 0, 0, 0, 0,+1, 0],
                [0,+1,+1,+1,+1,+1, 0],  
                [0,+1, 0, 0, 0, 0, 0],
                [0,+1, 0, 0, 0, 0, 0],
                [0,+1,+1,+1,+1,+1, 0],
                [0, 0, 0, 0, 0, 0, 0]])

    nine=np.array([ 
                [0, 0, 0, 0, 0, 0, 0],
                [0,+1,+1,+1,+1,+1, 0],
                [0,+1, 0, 0, 0,+1, 0],
                [0,+1, 0, 0, 0,+1, 0],
                [0,+1,+1,+1,+1,+1, 0],
                [0, 0, 0, 0, 0,+1, 0],
                [0, 0, 0, 0, 0,+1, 0],
                [0, 0, 0, 0, 0,+1, 0],
                [0, 0, 0, 0, 0, 0, 0]
                ])

    two = [2 * i - 1 for i in two]
    nine = [2 * i - 1 for i in nine]
    data = np.stack([two, nine])

    # ニューロン数
    n = 63
    # 記憶させたいパターン数
    k = 2
    # 学習データにどの程度ノイズを加えるか
    noise_rate = 0.1
    w = fit(n, data)
    flatten_two = data[1].flatten()
    test = noise(flatten_two, noise_rate)

    for i, v in enumerate(test):
        if i % 7 == 0 and i != 0:
            print('\n')
        print(f"{v:3d}", end="")
    print("\n")
    
    predicted = predict(test, w)
    predicted = np.array(predicted, dtype='int')
    print(type(predicted))
    
    for i, v in enumerate(predicted):
        if i % 7 == 0 and i != 0:
            print('\n')
        print(f"{v:3d}", end="")
    print("\n")

if __name__ == '__main__':
    main()