import scipy.optimize as opt


def f(z):
    x = z[0]
    y = z[1]
    return (x-2)**2+((y-1)*x)**2

def main():
    x0 = [3, 4]
    ans = opt.minimize(f, x0, method='Nelder-Mead')

    print(ans)
    print(2)

    
if __name__ == '__main__':
    main()

