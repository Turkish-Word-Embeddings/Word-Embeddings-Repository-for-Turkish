
class BaseLogger:

    def log(self, msg):
        raise NotImplementedError

class MockLogger:

    def log(self, msg):
        pass

class ConsoleLogger:

    def log(self, msg):
        print(msg)