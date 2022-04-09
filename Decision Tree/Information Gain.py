import numpy as np


def main():
    p = int(input("p: "))
    n = int(input("n: "))
    I = -p / (p + n) * np.log2(p / (p + n)) - n / (p + n) * np.log2(n / (p + n))
    print("I:", round(I, 3))


if __name__ == "__main__":
    main()
