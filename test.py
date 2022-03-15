def resolves():
    def wrapper(model):
        print('wrapper call')
        return model

    return wrapper


class A:
    def __set_name__(self, owner, name):
        print('set_name')


@resolves()
class C:
    a = A()


if __name__ == '__main__':
    print('main')
    c = C()
    print('end_main')
