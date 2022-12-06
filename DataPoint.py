class DataPoint:
    # what this class does is left as an exercise to the reader
    def __init__(self, inputs: list[float], evs: list[float]):
        self.inputs = inputs
        self.evs = evs

    def __str__(self):
        return str(self.inputs) + str(self.evs)
