import numpy as np

def run():
    #numpy scalar values can be represented as arrays with 0 dimensions
    scalar = np.array(5)
    print(scalar.shape)
    #numpy data types have unsigned and different sizes more similar to c types
    #numpy scalar types can have operations with python types
    print(type(scalar + 3))
    #one dimensional array shape is a one element tuple
    #can be sliced like python lists and have more powerful slicing
    onedimension = np.array([1, 3, 5])
    print(onedimension.shape)
    #two dimension is an array of arrays
    twodimension = np.array([[1,2],[1,2],[1,2]])
    print(twodimension.shape)
    #scalrs, vectors, two and higher dimension arrays are all tensors
    #the following is 2*3*2 tensor
    tensor = np.array([[[1,2],[1,2],[1,2]], 
                       [[1,2],[1,2],[1,2]]])
    print(tensor.shape)
    #reshape can change the matrix dimension without changing content
    onedimension2 = onedimension.reshape(1, 3)
    print(onedimension2.shape)
    #reshaping can be done with special numpy slicing syntax
    onedimension3 = onedimension[None, :]
    print(onedimension3.shape)
    #matrix operations can be done without iterations
    print(twodimension * 5)
    #squaring a matrix
    print(twodimension * twodimension)
    #resetting a matrix
    print(twodimension * 0)
    #numpy matrix dot product via matmul
    arr1 = np.array([[1, 2, 3],
                     [4, 5, 6]])
    arr2 = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]])
    dotproduct = np.matmul(arr1, arr2)
    print(dotproduct)

if __name__ == "__main__":
    run()
