import numpy as np
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
import multiprocessing as mp


def change_sensor(*args):
    existing_shm = shared_memory.SharedMemory(name='sensor')
    c = np.ndarray((6,), dtype=np.int64, buffer=existing_shm.buf)
    c[0] += 1
    print(c[0])
    existing_shm.close()

    existing_share_list = shared_memory.ShareableList(name='share_list')
    print(existing_share_list)


if __name__ == '__main__':
    a = np.array([1, 1, 2, 3, 5, 8], dtype=np.int64)
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes, name='sensor')
    shm.buf[:] = a.tobytes()
    shm_a = np.ndarray((6,), dtype=np.int64, buffer=shm.buf)
    print(shm_a)
    shlist = shared_memory.ShareableList([1, '2', 4], name='share_list')
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]
    a = b

    with mp.Pool(8) as p:
        p.map(change_sensor, [1, ] * 3)

    print(a)
    shm.close()
    shm.unlink()
