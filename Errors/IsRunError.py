class IsRunException(Exception):
    def __init__(self, message: str = ''):
        self.message = message

    def __str__(self) -> str:
        return f'{self.message}\nException: IsRunException'
