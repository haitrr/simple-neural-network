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
    #Run method
    #
    def Run(self,input):
        """Run the network base on the input data"""

        InCases = input.shape[0]

        #Clear out the previous intermediate value list
        self._layerInput = []
        self._layerOutput = []

        #Run it
        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T,np.ones([1,InCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,InCases])]))
            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sgm(layerInput))
        return self._layerOutput[-1].T


    #Transfer function
    def sgm(self,x,Derivative=False):
        if not Derivative:
            return 1/(1+np.exp(-x))
        else:
            out=self.sgm(x)
            return out*(1-out)


#
#If run as a script, create test object
#
if __name__ == "__main__":
    bpn = BackPropagationNetwork((2,2,2))
    print(bpn.shape)
    print(bpn.weights)
    lvInput = np.array([[0,0],[1,1],[-1,0.5]])
    lvOutput = bpn.Run(lvInput)
    print("Input : {0}\nOutput: {1}".format(lvInput,lvOutput))