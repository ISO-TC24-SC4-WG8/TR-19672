# -*- coding: utf-8 -*-

import math
import numpy as np
from tifffile import imsave
from skimage.draw import disk

def ComputePSD(particle_Sizes, x_min, x_max, nbins):
    
    Gauss_sample = np.log(particle_Sizes)

    # Gauss_bins = np.histogram_bin_edges(Gauss_sample, bins='auto')
    Gauss_bins = np.histogram_bin_edges(Gauss_sample, bins=nbins, range=(math.log(x_min), math.log(x_max)))
    bins = np.exp(Gauss_bins)
    
    # getting data of the histogram
    count, bins_count = np.histogram(particle_Sizes, bins)
      
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
      
    # using numpy np.cumsum to calculate the CDF
    cdf = np.cumsum(pdf)
    
    Q_0 = cdf
    X = bins[1:]
    
    InterValues = [.10, .16, .50, .84, .90, .99]
    
    for toInterpolateValue in InterValues:
        InterpolatedValue = np.interp(toInterpolateValue, Q_0, X)
        print("Q_0: ", toInterpolateValue, ", X: ", InterpolatedValue)
        
    return bins, X, Q_0  
        

class Particle:
    """A little class representing a sphere."""

    def __init__(self, cx, cy, cz, r, icolour=None):
        """Initialize the particle with its center, (cx,cy) and radius, r.
        """
        self.cx, self.cy, self.cz, self.r = cx, cy, cz, r

    def overlap_with(self, cx, cy, cz, r):
        """Does the circle overlap with another of radius r at (cx, cy, cz)?"""
        
        # euclidian distance between center points
        d = math.sqrt((cx-self.cx)**2 + (cy-self.cy)**2 + (cz-self.cz)**2 )
        return d < r + self.r
    

class Particles:
    """A class for drawing particles"""

    def __init__(self, width=2048, height=2048, depth=1000, n=5, x_median=50,
                 x_sigma=2.5, monodisperse=False, colours=None):
        """Initialize the particles object lists."""
        self.particles_sizes = []
        self.boundary_particles = []
        self.width, self.height, self.depth = width, height, depth
        self.n = n
        # The center of the generation space
        # self.CX, self.CY = self.width // 2, self.height // 2
        self.r_median, self.sigma_r = x_median/2, x_sigma
        self.monodisperse=monodisperse


    def print_particles(self, image, ForegroundBrightness, *args, **kwargs):
                     
        img_height,img_width = image.shape
        
        for Particle in self.particles:
  
            #does the particle center lie within the image (dimensions)?                                             
            if Particle.cx >= 0 and Particle.cy >= 0 and Particle.cx <= img_width and Particle.cy <= img_height:  

                ### For circular particles ########                
                rr, cc = disk((Particle.cx, Particle.cy), Particle.r, shape=image.shape)                                                    
                image[rr, cc] = ForegroundBrightness
                self.particles_sizes.append(Particle.r*2)
                # #####################################
                        
                # does the particle touch the the image boundaries (incomplete boundary  particle)?
                # TODO resolve the difference to the boundary check in _place_particle().
                # Define: Where is the coordinate of a pixel? at the center or at the lower left corner of the pixel (preferred)
                if int(Particle.cx - Particle.r + .5) <= 0 or int(Particle.cy - Particle.r + .5) <= 0 or int(Particle.cx + Particle.r +.5) >= img_width or int(Particle.cy + Particle.r + .5) >= img_height:
                    self.boundary_particles.append(True)
                else:
                    self.boundary_particles.append(False)
                    

    def _place_particle(self, r, image, excludeBorderParticles, MinSpacing):
        """Place the particles inside the generation volume."""
        img_height,img_width = image.shape
        # The trials number: if a particle with a given size is not succesfully placed within this number
        # of trials, the placement process is stopped .
        trials = 1000
        while trials:
            # Pick a random position, uniformly in the sampling volume.
            # Consequences:
            # - particles having their center outside the volume are not generated even
            #   if they extend beyond the border of the picture into the picture.
            # - the generation volume is the same vor all particles
            # - all particles generated are also drawn, unless border particles are excluded
            cx = self.width*np.random.random()
            cy = self.height*np.random.random()
            cz = self.depth*np.random.random()
            # check if the particle touches the image boundaries
            if cx - (r + MinSpacing) <= 0 or cy - (r + MinSpacing) <= 0  or cx + (r + MinSpacing) >= img_width or cy + (r + MinSpacing) >= img_height:
                boundary_particle = True
            else:
                boundary_particle = False
            
            if (excludeBorderParticles and not boundary_particle) or (not excludeBorderParticles):
                # Check if the particle (as a volume, sphere) overlaps with a particle alreadey in the list.
                if not any(particle.overlap_with(cx, cy, cz, r + MinSpacing)
                           for particle in self.particles):
                    # The particle doesn't overlap any other particle: place it.
                    particle = Particle(cx, cy, cz, r)
                    self.particles.append(particle)
                    return
            trials -= 1
        # Warn that the upper limmit of attempts was reached
        print('Warning: limit of trials reached. The current particle is being excluded')

    def generate_particles(self, image, excludeBorderParticles, minSpacing):
        """Generate a new set of particles for each image following the given size distribution"""
        np.random.seed()
        # randomize the number of particles per image
        npp = np.random.poisson(self.n)
        self.particles = [] # start with a new list
        
        # create a vector containing npp number of radius values
        if self.monodisperse==False:
            ## Transform normal into log-normal und generate particle size array "r"
            mu, sigma = np.log(self.r_median), np.log(self.sigma_r)
            r = np.exp(np.random.normal(mu, sigma, npp))
        else:
            r =np.full(npp,self.r_median)
        
        # placing the particles
        for i in range(npp):
            self._place_particle(r[i], image, excludeBorderParticles, minSpacing)
            
##############################################################################
##############################################################################
## Main Code Source
##############################################################################
##############################################################################

# Particles are randomly generated within the sampling volume.
# The particles placed are not allowed to overlap with each other within the volume.
     
# Image Frame Size (Field of View), pixel units
FOVHeight = 2048
FOVWidth = 2048
# Sampling Volume Depth, pixel units
SVDepth = 200

BackgroundBrightness = 255
ForegroundBrightness = 0 # color of the particles

# Normal distribution of the particle sizes (by number) in pixel units
x_stdev =  2.76/2.0  # Px
x_median = 20 # Px
StrictMonodisperse = False # if set to True, stdev_x is ignored

# Drawing parameters
NumOfFrames = 500 # number of frames
NumOfParticlesPerImage = 400 # average number of particles per generation volume
excludeBorderParticles = False # True -> do not draw particles crossing the edges of the field of view
MinSpacing = 2 # allowing for a minimum space between the particles (3D)

# Pixel size and magnification
# In order to compare particle size results calculated by this script in
# micrometers with other software or instruments, provide the correct scaling factors 
PixelWidth = 5.5 # unit µm
invMagnification = 1.0  
effPixelWidth = invMagnification * PixelWidth  

# Histogramm settings
NumberofSizeClasses = 90
x_min = effPixelWidth
x_max = max(FOVHeight, FOVWidth)*effPixelWidth

# Create the particle drawing object
particles = Particles(width=FOVWidth, height=FOVHeight, depth=SVDepth, n=NumOfParticlesPerImage, x_median=x_median, x_sigma=x_stdev, monodisperse = StrictMonodisperse)

# Generate the particle images
np.random.seed()
for j in range(NumOfFrames): 
    # Create the empfty image array
    image = np.empty((FOVHeight,FOVWidth), 'uint8')
    image[:,:] = BackgroundBrightness
    # Generate and print the particles image by image 
    particles.generate_particles(image, excludeBorderParticles, MinSpacing)       
    particles.print_particles(image, ForegroundBrightness) 

    # Collect all images into an image stack
    print("Frame: ",j + 1)
    if(j==0):
        image_stack = image # create the stack from the first image
    else:
        image_stack = np.dstack((image_stack,image))

# Save image stack as TIFF
# The creation of the full image stack in memory limits the number of images. 
# It may be preferred not to create the stack and output the images directly
# one by one.
image_stack = np.transpose(image_stack, (2,0,1))
imsave('image_stack.tif', image_stack, compression='zlib') #, compressionargs={'level': 8}, predictor=True)
  
print("\neffPixelWidth: ", effPixelWidth)

# Display a single image as an example
from matplotlib import pyplot as plt
plt.imshow(image_stack[1], cmap='gray')
plt.show()

if StrictMonodisperse == False:
    particle_Sizes = np.array(np.asarray(particles.particles_sizes), 'float64')*effPixelWidth
    boundary_Particles = particles.boundary_particles
    
    print("\nPSD's p-values of all particles on picture:")
    bins, X, Q_0  = ComputePSD(particle_Sizes, x_min, x_max, NumberofSizeClasses)
    
    NumOfNonBorderParticles = 0
    for boundaryParticle in particles.boundary_particles:   
        if not boundaryParticle:
            NumOfNonBorderParticles = NumOfNonBorderParticles+1
            
    nonborderParticle_Sizes = np.zeros(NumOfNonBorderParticles)
    
    k=0
    j=0
    for boundaryParticle in particles.boundary_particles:   
        if not boundaryParticle:
            nonborderParticle_Sizes[k] = particle_Sizes[j]
            k = k+1
        j = j+1

    print("\nNumber of particles: ", len(particle_Sizes))
    print("\nNumber of non border particles: ", len(nonborderParticle_Sizes))

    print("\nPSD's p-values of all non-border particles:")
    bins_non, X_non, Q_0_non  = ComputePSD(nonborderParticle_Sizes, x_min, x_max, NumberofSizeClasses)

    plt.semilogx(X, Q_0)
    plt.semilogx(X_non, Q_0_non)