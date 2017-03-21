"""
Description: The following is an implementation of the Eigenfaces - algorithm. 
This algorithm is used to for face recognition and is a PCA based implementation.
Further information regarding how to implement this algorithm can be found
at the following resource.

For data - The AT&T database of faces was used.

The EigenFaces Method is essentially PCA with a Nearest Neighbor Model. 
"""

# Import Declarations
import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.cm as cm
import sys

# Functions
def read_images(filePath, # File Path 
				sz=None):  # Cropped Image Size
	#Program vars
	
	# Image count
	imageCount = 0
	x,y = [], []

	# Computation
	
	# Path is the top level file path where images are contained
	for  dirname , dirnames , filenames  in os.walk(path):
		for  subdirname  in  dirnames:
			subject_path = os.path.join(dirname , subdirname)
			for  filename  in os.listdir(subject_path):
				try:
					im = Image.open(os.path.join(subject_path , filename))
					im = im.convert("L")
					# resize  to  given  size (if  given)
					if (sz is not  None):
						im = im.resize(sz, Image.ANTIALIAS)
		
					x.append(np.asarray(im , dtype=np.uint8))
					y.append(c)

				# Handle IO error
				except  IOError:
					print "I/O error ({0}): {1}".format(errno , strerror)
				# For any other exeption
				except:
					print "Unexpected  error:", sys.exc_info ()[0]
					raise
		imageCount += 1
	return [x,y]

def  asRowMatrix(X):
	if len(X) == 0:
		return  np.array ([])
	mat = np.empty ((0, X[0]. size), dtype=X[0]. dtype)
	for  row in X:
		mat = np.vstack ((mat , np.asarray(row).reshape (1,-1)))
	return  mat

def  asColumnMatrix(X):
	if len(X) == 0:
		return  np.array ([])
	# else
	mat = np.empty ((X[0].size , 0), dtype=X[0]. dtype)
	for  col in X:
		mat = np.hstack ((mat , np.asarray(col).reshape (-1,1)))
	return  mat

# Implementing PCA - Principal Component Analysis
# The implmenetation follows the steps in the paper described in comments
def  pca(X, # random vector of feature observations
		 y, 
		 num_components =0):
	# determine order of random vector
	[n,d] = X.shape
	if (num_components  <= 0) or (num_components >n):
		num_components = n
	# computing the mean	
	mu = X.mean(axis =0)
	# subtract mean vector
	X = X - mu
	if n > d:
		C = np.dot(X.T,X)
		# compute eigenvalues and eigenvectors
		[eigenvalues ,eigenvectors] = np.linalg.eigh(C)
	else:
		# n <= d
		C = np.dot(X,X.T)
		# compute eigenvalues and eigenvectors
		[eigenvalues ,eigenvectors] = np.linalg.eigh(C)
		eigenvectors = np.dot(X.T,eigenvectors)
	
	for i in  xrange(n):
		eigenvectors [:,i] = eigenvectors [:,i]/np.linalg.norm(eigenvectors [:,i])

	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors [:,idx]
	# select  only  num_components
	eigenvalues = eigenvalues [0: num_components ].copy()
	eigenvectors = eigenvectors [:,0: num_components ].copy()
	return [eigenvalues , eigenvectors , mu]



