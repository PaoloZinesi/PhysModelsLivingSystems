import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import PIL.Image as Image 
import matplotlib.image as img
import sys, getopt

# optional control on verbosity
verbose = True if (sys.argv[3]=="T" or sys.argv[3]=="True") else False

# read an image and convert it to a binary pattern os size [Lx, Ly]: 
# arguments = figurename, final size=[Lx, Ly]
def readPatterns(fname, size, show_img=True):
	this_img = Image.open(fname) # open colour image
	this_img = this_img.convert('1') # convert image to black and white
	this_img = this_img.resize(size=[size[0],size[1]]) # resize it to have the dimensions [Lx, Ly]
	this_img.save(f"{fname[:-4]}_converted.png") # save converted image

	if show_img:
		# plot the original and the converted file
		fig = plt.figure()

		# subplot on the left - original figure
		fig.add_subplot(1,2,1)		
		original = img.imread(fname)
		plt.imshow(original)
		plt.title("original")
		plt.axis('off')

	# subplot on the right - converted figure
	matrix = img.imread(f"{fname[:-4]}_converted.png") # re-read from the file and convert it to a matrix of [0.,1.]
	matrix = 2*matrix-1 # convert from [0.,1.] -> [-1,1]

	if show_img:
		fig.add_subplot(1,2,2)		
		plt.matshow(matrix, cmap=plt.cm.gray,fignum=0)
		plt.axis('off')
		plt.title("simplified")
		plt.show()
	
	# Return the pattern as a 1D vector (not as a 2D a matrix)
	return matrix.flatten()
	
#####################################################################################



class HopfieldNet:
	# initialize: 
	# arguments = number of neurons, list of patterns (vector of M components, each element of the pattern has to be an array of -1,+1 of size N)
	def __init__(self, N, patterns):
		self.N = N
		self.time_elapsed = 0.
		
		self.w = np.zeros([N,N]) # weights
		self.h = np.zeros(N) # threshold functions
		
		self.s = -np.ones(N) # default configuration = s[i]=-1
		
		# HEBBIAN RULE (h_i = 0., w_{ij} = sum_{k=1,...,M} s_i^k*s_j^k / M)
		if(verbose): print("The network is learning...")
		self.M = len(patterns)
		for k in range(self.M):
			if(verbose): print("pattern ", k)
			# this is not efficient, but we could use it anyway:
#			for i in range(self.N):
#				for j in range(self.N):
#					self.w[i,j] += patterns[k][i]*patterns[k][j]/(1.*self.M)

			# it is more efficient to use built-in functions:
			self.w += np.outer(patterns[k],patterns[k])/(1.*self.M)


		if(verbose): print("Done!")
	
	
#		# COMPUTE THE ENERGY - As before, I avoid loops and use efficient functions
		self.E = -0.5*np.sum(self.w) - np.sum(self.h) # energy for s_i = -1
		
		return

	# given and input s=[s_1,s_2,...,s_N], set the state of the network and recompute the energy
	def set_state(self, sinput):
		self.s = np.copy(sinput)

		# COMPUTE THE ENERGY - I use efficient functions rather than loops
		s2 = np.outer(self.s, self.s) # this returns a matrix s2[i,j]=s[i]*s[j]
		self.E = -0.5*np.sum(self.w*s2) + np.sum(self.h*self.s)
	
		return

	# evolve the state of the networks doing a number "steps" of Monte Carlo steps
	def evolve(self, steps):
		for t in range(steps):
			i = np.random.randint(self.N) # choose one node randomly
			
			sum_wijsj = np.sum(self.w[i,:]*self.s) # compute the argument of the activation function			
			if sum_wijsj < self.h[i]: # below the threshold
				self.s[i] = -1
			else: # above the threshold
				self.s[i] = 1
				
		return
#######################################################################


# dimensions of the images
Lx = Ly = 75
N = Lx*Ly # number of neurons



## STEP 1: READ THE IMAGES AND CONVERT THEM TO BINARY PATTERNS

# list of images that I want to store in my network
files = ["stored/batman.png", "stored/cat.jpg", "stored/jordi.jpg"]

if(verbose): print("Reading images and converting to binary patterns...")
patterns = []
for fname in files:
	patterns.append(readPatterns(fname, size=[Lx,Ly], show_img=verbose))
if(verbose): print("Done!")


## STEP 2: CREATE THE NETWORK AND LEARN THE PREVIOUS PATTERNS
mynet = HopfieldNet(N, patterns)


## STEP 3: SET ANOTHER INPUT PATTERN AS INITIAL CONDITION
newinput = readPatterns(sys.argv[1], size=[Lx,Ly], show_img=False) # read the pattern from a file GIVEN AS FIRST ARGUMENT IN COMMAND-LINE
mynet.set_state(newinput) # set the pattern as the initial condition of the network 
plt.matshow(np.resize(mynet.s,(Lx,Ly)), cmap=plt.cm.gray, fignum=0) # read the state (we reconvert it to a Lx*Ly matrix) and plot it
plt.title("Initial condition")
plt.axis('off')
plt.show()

## STEP 4: EVOLVE THE NETWORK!
#################### ANIMATION #################################
# animation
TOTAL_FRAMES = 1000 # number of frames of the animation
DT_FRAMES = 20 # duration of each frame (ms): we can accelerate or slow down the animation changing this

fig = plt.figure()	
fig_mat = plt.matshow(np.resize(mynet.s,(Lx,Ly)), cmap=plt.cm.gray, fignum=0) # see more colormaps in https://matplotlib.org/examples/color/colormaps_reference.html
plt.axis('off')

# functions used to produce the animation
def animation_init(): # how to initialize the animation (if we put things here they will be fix in the animation) 
	fig_mat.set_data(np.resize(mynet.s,(Lx,Ly)))
	return fig_mat

def animation_update(i):
	mynet.evolve(steps=mynet.N//10)
	fig_mat.set_data(np.resize(mynet.s,(Lx,Ly)))
	return fig_mat
	
# show animation only if block_animation=="True" (or "T")
block_animation = True if (sys.argv[2]=="T" or sys.argv[2]=="True") else False
if not block_animation:
	# generate the animation
	ani = animation.FuncAnimation(fig, animation_update, frames=TOTAL_FRAMES, interval=DT_FRAMES, blit=False, repeat=False) # blit is used to make simulation faster
	plt.show()
else:
	# evolve the network without showing the animation
	for i in range(TOTAL_FRAMES):
		animation_update(i)


plt.matshow(np.resize(mynet.s,(Lx,Ly)), cmap=plt.cm.gray, fignum=0)
plt.title("Final state")
plt.axis('off')
plt.show()








