
import scipy

def test(one, **kwargs):
    print(one)
    check = input('Put into')
    
    return check

if __name__ == '__main__':
    test('testing', ex='one', two='two', three='three')