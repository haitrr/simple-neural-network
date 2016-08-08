#
#Import
#
import numpy as np


# Transfer function
def sgm(x, Derivative=False):
    if not Derivative:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        out = sgm(x)
        return out * (1.0 - out)

def linear(x, Derivative=False):
    if not Derivative:
        return x
    else:
        return 1.0
def gausian(x,Derivative = False):
    if not Derivative:
        return np.exp(-x**2)
    else:
        return -2*x*np.exp(-x**2)
def tanh(x,Derivate = False):
    if not Derivate:
        return np.tanh(x)
    else:
        return 1.0 - np.tanh(x)**2
#
#Classes
#

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
    def __init__(self,layerSize,layerFunction = None):
        """Initialize the network"""

        #layer info
        self.layerCount=len(layerSize) -1
        self.shape = layerSize

        if layerFunction is None:
            lFuncs= []
            for i in range(self.layerCount):
                if i == self.layerCount + 1:
                    lFuncs.append(linear)
                else:
                    lFuncs.append(sgm)
        else:
            if len(layerSize) != len(layerFunction):
                raise ValueError("Incompatible list of tranfer functions.")
            elif layerFunction[0] is not None:
                raise ValueError("Input layer can not have a layer function")
            else:
                lFuncs=layerFunction[1:]

        self.tFuncs=lFuncs

        #Data from the last run
        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []


        #Create the weight arrays
        for(l1,l2) in zip(layerSize[:-1],layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.1,size= (l2,l1+1)))
            self._previousWeightDelta.append(np.zeros((l2,l1-1)))


    #
    #Run method
    #
    def Run(self,input):
        """Run the network base on the input data"""

        lnCases = input.shape[0]

        #Clear out the previous intermediate value list
        self._layerInput = []
        self._layerOutput = []

        #Run it
        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T,np.ones([1,lnCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,lnCases])]))
            self._layerInput.append(layerInput)
            self._layerOutput.append(self.tFuncs[index](layerInput))
        return self._layerOutput[-1].T

    #
    #TrainEpoch method
    #
    def TranEpoch(self,input,target,trainingRate = 0.2,momentum = 0.5):
        """This method train the network for one epoch"""

        delta = []
        lnCase = input.shape[0]

        # Run the network first
        self.Run(input)

        #
        #Calculate delta
        #
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                #Compare to the target values
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta * self.tFuncs[index](self._layerInput[index],True))
            else:
                #Compare to the following layer's delta
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1,:]*self.tFuncs[index](self._layerInput[index],True))
        #Compute weight deltas
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index
            if index == 0:
                layerOutput = np.vstack([input.T,np.ones([1,lnCase ])])
            else:
                layerOutput = np.vstack([self._layerOutput[index - 1],np.ones([1,self._layerOutput[index - 1].shape[1]])])
            curWeightDelta = np.sum(
                layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0)
                ,axis = 0)
            weightDelta = trainingRate * curWeightDelta + momentum * self._previousWeightDelta[index]
            self.weights[index] -= weightDelta

            self._previousWeightDelta[index] = weightDelta
        return error

#
#If run as a script, create test object
#
if __name__ == "__main__":

    lvInput = np.array([[0,0],[1,1],[0,1],[1,0]])
    lvTarget = np.array([[0.00],[0.00],[1.00],[1.00]])
    lFuncs = [None,gausian,tanh,linear]
    bpn = BackPropagationNetwork((2,2, 2, 1),lFuncs)
    lnMax = 100000
    lnError = 0.00001
    for i in range(lnMax + 1):
        err = bpn.TranEpoch(lvInput,lvTarget,momentum=0.7)
        if i % 2500 == 0:
            print("Iteration {0}\tError: {1:0.6f}".format(i,err))
        if err <= lnError:
            print ("Minimum error reached at irration {0}".format(i))
            break

    #Dislay output
    lvOutput = bpn.Run(lvInput)
    for i in range(lvInput.shape[0]):
        print("Input : {0}  Output : {1}".format(lvInput[i],lvOutput[i]) )