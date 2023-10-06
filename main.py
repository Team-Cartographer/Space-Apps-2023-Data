class CodeNotWrittenError(Exception):
    def __init__(self, message="This code is not implemented yet"):
        self.message = message
        super().__init__(self.message)

if __name__ == "__main__":
    raise CodeNotWrittenError()
  
