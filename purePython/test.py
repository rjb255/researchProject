from pathos.multiprocessing import ProcessPool as Pool
from multiprocessing import Queue, Process
import time


def a(x, q):
    time.sleep(4)
    q.put(x + 1)


def f(x):
    return x**2


def g(x):
    return x**3


def main():
    t0 = time.time()

    def getResults(algorithm):
        score = []
        processes = []
        for i in range(3):
            s = algorithm(i)
            score.append(Queue())
            processes.append(Process(target=a, args=(s, score[-1])))
            processes[-1].start()

        print(algorithm.__name__)
        return [s.get() for s in score]

    with Pool() as p:
        results = p.map(getResults, (f, g))
    t1 = time.time()
    print(f"time: {t1-t0}")  # ~4.5s
    print(list(results))

    # [[1, 2, 5, 10, 17, 26, 37, 50, 65, 82],
    # [1, 2, 9, 28, 65, 126, 217, 344, 513, 730]]


if __name__ == "__main__":
    main()
