#
#Import
#
import numpy as np

class BackPropagationNetwork:
    """A back-propagation network"""
    #
    #Class member
    #
    layerCount = 0
    shape = None
    weights = []

    #
    #Class methods
    #
    def __init__(self,layerSize):
        """Initialize the network"""

        #layer info
        self.layerCount=len(layerSize) -1
        self.shape = layerSize

        #Input/Ouput data from the last run
        self._layerInput = []
        self._layerOutput = []

        #Create the weight arrays
        for(l1,l2) in zip(layerSize[:-1],layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.1,size= (l2,l1+1)))


#
#If run as a script, create test object
#
if __name__ == "__main__":
    bpn = BackPropagationNetwork((2,2,1))
    print(bpn.shape)
    print(bpn.weights)