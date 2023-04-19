from multiprocessing import Manager


def

if __name__ == '__main__':

    a = Manager().list()
    for i in range(10):
        a.append((i, str(i)))
    print(a)
