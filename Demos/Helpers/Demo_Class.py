from abc import abstractmethod

class Demo():
    def __init__(self) -> None:
        pass

    def get_name(self):
        return self.__class__.__name__
    
    @abstractmethod
    def do(self):
        print("do, pls")
        pass
