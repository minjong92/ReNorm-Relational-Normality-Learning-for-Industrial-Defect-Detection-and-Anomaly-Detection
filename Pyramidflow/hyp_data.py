class MHyp:

    def __init__(self):
        self.epoches = 15
        self.encoder = 'resnet18'
        self.numLayer = 4
        self.volumeNorm = 'CVN'
        self.kernelSize = 7
        self.numChannel = 16
        self.numStack = 4
        self.gpu = 0
        self.batchSize = 4

        self.brightness = 0.3
        self.contrast = 0.3
        self.saturation = 0.3
        self.hue = 0.3
        self.hflip = 0.5
        self.vflip = 0.5
        self.rotate90 = 0.5
        self.valRatio = 0.1

        self.lossMode = 0
        self.continuityAlpha = 0.25
        self.continuityBeta = 0.5
        self.continuityMode = 'l1'

        self.pixelThresholdPercentile = 99.5
        self.imageThresholdPercentile = 95.0

    def print_data(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")

class MData:

    def __init__(self):
        pass

    def print_data(self):
        for key, value in self.__dict__.items():
            print(f"{key}: {value}")
