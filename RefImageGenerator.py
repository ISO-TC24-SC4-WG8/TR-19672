# -*- coding: utf-8 -*-
import math
import numpy as np
from tifffile import imsave, TiffWriter
from skimage.draw import disk, ellipse
from myskidraw import superellipse
from scipy.special import gamma


def ComputePSD(particle_sizes, x_min, x_max, nbins):
    """Auxiliary function for calculating a particle size distribution Q0.

    Parameter
    ----------
    particle_sizes : array
        array of all particles sizes, one entry for each particle

    Output
    ----------
    Prints interpolated values for the percentiles x_10, x_16, x_50, x_84, x_90.

    Returns
    ----------
    bins : array
        array of the histogram counts
    X : array
        array of the upper interval boundaries of the histogram
    Q_0 : array
        array of the cumulative distribution by number
    """
    # Input data for the subsequent histogram_bin_edges function.
    # This is used by this function for several auto-estimation features
    # of bins. At the moment only nbins with equal-width bins are calulated
    log_particle_sizes = np.log(particle_sizes)

    # Function to calculate only the edges of the nbins equal-width bins
    # used by the histogram function.
    log_bins = np.histogram_bin_edges(
        log_particle_sizes, bins=nbins, range=(math.log(x_min), math.log(x_max)))

    # back to the real values for the sizes
    bins = np.exp(log_bins)

    # getting data of the histogram
    count, bins_edges = np.histogram(particle_sizes, bins)

    # using numpy np.cumsum to calculate the cumulative distribution
    Q_0 = np.cumsum(count / sum(count))

    X = bins[1:]

    InterValues = [.10, .16, .50, .84, .90, .99]

    for toInterpolateValue in InterValues:
        # this interpolates on a linear scale X - TODO change to logscale
        InterpolatedValue = np.interp(toInterpolateValue, Q_0, X)
        print("Q_0: ", toInterpolateValue, ", X: ", InterpolatedValue)

    return bins, X, Q_0


class Particle:
    """A base class representing a particle.
    This basic particle carries its geometrical center, (cx,cy,cz),
    and radius of a minimum circumscribed sphere _around_ this center, r.
    This information is only used to check the mutual overlap in the measurement
    volume and the overlap to the edge of the volume.
    """

    def __init__(self, r, center):
        """Initialize the particle with its geometrical center, cx,cy,cz,
        and with a radius of a minimum circumscribed sphere _around_ this
        center, r.

        Parameter
        ----------
        r : float
            radius of a minimum circumscribed sphere _around_ the center
        center : tuple (cx,cy,cz) of float
            geometrical center of the particle in the measurement volume
            This is also the centre of mass for all shapes simulated
            (symmetrical particles).

        Returns
        ----------
        the particle object
        """
        self.r = r
        self.cx, self.cy, self.cz = center

    def overlap_with(self, r, cx, cy, cz):
        """Does the particle overlap with another of radius r at (cx, cy, cz)
        in 3D? This check uses the euclidian distance and is exact only for
        spheres and used as a simplified check for other type of shapes.

        TODO Change logic, because overlapp cannot be detected exactly,
        but having no overlapp can.

        Parameter
        ----------
        r : float
            radius of a minimum circumscribed sphere _around_ the center
        cx,cy,cz : float
            geometrical center of the particle position to be checked

        Returns
        ----------
        True|False : bool
            overlapp may occur?
        """

        # calculate the euclidian distance between center points
        d = math.sqrt((cx-self.cx)**2 + (cy-self.cy)**2 + (cz-self.cz)**2)
        # if the distance is smaller than the sum of the enclosing radii
        # the particles may overlapp. Exact for spheres. Only a workaround for
        # other shapes
        return d < r + self.r

class SphericalParticle(Particle):
    """class representing a spherical particle (3D)
    since the shape is defined, various size parameters can be set.
    The object carries all size parameters that can be derived from the diameter
    x. The sphere does not carry an orientation.
    """
    def __init__(self, x, center):
        """Initialize the spherical particle with its location, cx,cy,cz, and
        diameter.

        Parameter
        ----------
        x : float
            diameter of the sphere
        center : tuple (cx,cy,cz) of float
            geometrical center of the particle in the measurement volume
            This is also the centre of mass for all shapes simulated
            (symmetrical particles).

        Returns
        ----------
        the SphericalParticle object
        """
        # init parent with center location and minimum circumscribed radius
        super().__init__(x/2.0, center)
        # Definitions from ISO9276-6, size parameters
        # Volume V
        self.size_V=4.0/3.0*math.pi*math.pow(x/2.0, 3.0)
        # Projection area A
        self.size_A=math.pi*math.pow(x/2.0, 2.0)
        # Area equivalent diameter x_A
        self.size_x_A=x
        # Volume equivalent diameter x_V
        self.size_x_V=x
        # Feret Diameters x_Fmax, x_Fmin
        self.size_x_Fmax=x
        self.size_x_Fmin=x
        # Ellipse Axes x_Lmax, x_Lmin
        self.size_x_Lmax=x
        self.size_x_Lmin=x
        # Perimeter P
        self.size_P=x*math.pi

    def draw(self, image, ForegroundBrightness):
        # TODO generalize the overlapp check into the base class
        img_height,img_width = image.shape

        self.isdrawn=False
        #does the particle center lie within the image (dimensions)?
        if self.cx >= 0 and self.cy >= 0 and self.cx <= img_width and self.cy <= img_height:
            ### For circular particles ########
            rr, cc = disk((self.cx, self.cy), self.r, shape=image.shape)
            image[rr, cc] = ForegroundBrightness
            self.isdrawn=True

            # self.particles_sizes.append(Particle.r*2)

            # does the particle touch the the image boundaries (incomplete boundary  particle)?
            # TODO resolve the difference to the boundary check in _place_particle().
            # Define: Where is the coordinate of a pixel? at the center or at the lower left corner of the pixel (preferred)
            if int(self.cx - self.r + .5) <= 0 or int(self.cy - self.r + .5) <= 0 or int(self.cx + self.r +.5) >= img_width or int(self.cy + self.r + .5) >= img_height:
                # self.boundary_particles.append(True)
                self.isboundary = True
            else:
                # self.boundary_particles.append(False)
                self.isboundary = False

class EllipseParticle(Particle):
    """class representing an elliptical particle (2D)
    since the shape is defined, various size parameters can be set.
    The object carries all size parameters that can be derived from the size
    given during initialisation (size is given as ecd).
    """
    def __init__(self, x_major, x_minor, center, rotation=0):
        """Initialize the ellipse with its location, cx,cy,cz, and axes.

        Parameter
        ----------
        x_major, x_minor : float
            major and minor full axes of the ellipse
        center : tuple (cx,cy,cz) of float
            geometrical center of the particle in the measurement volume
            This is also the centre of mass for all shapes simulated
            (symmetrical particles).
        rotation : float
            rotation of the ellipse, in radians in range (-PI, PI),
            in contra clockwise direction, with respect to the axis.

        Returns
        ----------
        the EllipseParticle object
        """
        # init parent with center location and minimum circumscribed radius
        super().__init__(x_major/2.0, center)
        self.rotation=rotation
        # Definitions from ISO9276-6, size parameters
        # Volume V - a 2D shape has no volume
        self.size_V=None
        # Projection area A
        self.size_A=math.pi*x_major/2*x_minor/2
        # Area equivalent diameter x_A
        self.size_x_A=math.sqrt(4*self.size_A/math.pi)
        # Feret Diameters x_Fmax, x_Fmin
        self.size_x_Fmax=x_major
        self.size_x_Fmin=x_minor
        # Ellipse Axes x_Lmax, x_Lmin
        self.size_x_Lmax=x_major
        self.size_x_Lmin=x_minor
        # Perimeter P - no simple formula - via elliptic integral
        # self.size_P=None

    def draw(self, image, ForegroundBrightness):
        img_height,img_width = image.shape

        self.isdrawn=False
        #does the particle center lie within the image (dimensions)?
        if self.cx >= 0 and self.cy >= 0 and self.cx <= img_width and self.cy <= img_height:
            ### For circular particles ########
            rr, cc = ellipse(self.cx, self.cy, self.size_x_Lmax/2, self.size_x_Lmin/2, shape=image.shape, rotation=self.rotation)
            image[rr, cc] = ForegroundBrightness
            self.isdrawn=True

            # self.particles_sizes.append(Particle.r*2)

            # does the particle touch the the image boundaries (incomplete boundary  particle)?
            # TODO resolve the difference to the boundary check in _place_particle().
            # Define: Where is the coordinate of a pixel? at the center or at the lower left corner of the pixel (preferred)
            if int(self.cx - self.r + .5) <= 0 or int(self.cy - self.r + .5) <= 0 or int(self.cx + self.r +.5) >= img_width or int(self.cy + self.r + .5) >= img_height:
                # self.boundary_particles.append(True)
                self.isboundary = True
            else:
                # self.boundary_particles.append(False)
                self.isboundary = False

class SuperEllipseParticle(Particle):
    """class representing an superelliptical particle (2D)
    since the shape is defined, various size parameters can be set.
    The object carries all size parameters that can be derived from the size
    given during initialisation (size is given as ecd).
    """
    def __init__(self, x_major, x_minor, exponent, center, rotation=0):
        """Initialize the superellipse.

        Parameter
        ----------
        x_major, x_E : float
            major and minor full axes of the superellipse, here interpreted
            as geodesic length and thickness
        exponent : float
            The overall shape of the superellipse is determined by the value
            of the exponent
        center : tuple (cx,cy,cz) of float
            geometrical center of the particle in the measurement volume
            This is also the centre of mass for all shapes simulated
            (symmetrical particles).
        rotation : float
            rotation of the ellipse, in radians in range (-PI, PI),
            in contra clockwise direction, with respect to the axis.

        Returns
        ----------
        the SuperEllipseParticle object
        """
        super().__init__(x_major/2.0, center)
        self.rotation=rotation
        self.exponent=exponent
        # Definitions from ISO9276-6, size parameters
        # Volume V - a 2D shape has no volume
        # TODO not adapted to a super ellipse!
        self.size_V=None
        # Projection area A
        # The area inside the superellipse can be expressed in terms of
        # the gamma function
        self.size_A=x_major*x_minor*math.pow(gamma(1+1/exponent),2)/gamma(1+2/exponent)
        # Area equivalent diameter x_A
        self.size_x_A=math.sqrt(4*self.size_A/math.pi)
        # Feret Diameters x_Fmax, x_Fmin
        #self.size_x_Fmax=x_major
        #self.size_x_Fmin=x_minor
        # Ellipse Axes x_Lmax, x_Lmin
        #self.size_x_Lmax=x_major
        #self.size_x_Lmin=x_minor
        # geodesic length x_LG and thickness x_E
        self.size_x_LG=x_major
        self.size_x_E=x_minor
        # Perimeter P - no simple formula
        # self.size_P=None

    def draw(self, image, ForegroundBrightness):
        img_height,img_width = image.shape

        self.isdrawn=False
        #does the particle center lie within the image (dimensions)?
        if self.cx >= 0 and self.cy >= 0 and self.cx <= img_width and self.cy <= img_height:
            ### For circular particles ########
            rr, cc = superellipse(self.cx, self.cy, self.size_x_LG/2, self.size_x_E/2, self.exponent, shape=image.shape, rotation=self.rotation)
            image[rr, cc] = ForegroundBrightness
            self.isdrawn=True

            # self.particles_sizes.append(Particle.r*2)

            # does the particle touch the the image boundaries (incomplete boundary  particle)?
            # TODO resolve the difference to the boundary check in _place_particle().
            # Define: Where is the coordinate of a pixel? at the center or at the lower left corner of the pixel (preferred)
            if int(self.cx - self.r + .5) <= 0 or int(self.cy - self.r + .5) <= 0 or int(self.cx + self.r +.5) >= img_width or int(self.cy + self.r + .5) >= img_height:
                # self.boundary_particles.append(True)
                self.isboundary = True
            else:
                # self.boundary_particles.append(False)
                self.isboundary = False

class PShape:
    """A base class representing a particle shape without assigning a size."""
    def __init__(self, ratio):
        """Initialisation of a pasic shape having a proportion descriptor
        of any kind

        Parameter
        ----------
        ratio : float
            ratio of a the proportion descriptor with values between 0 and 1
        """
        self.ratio=ratio

class PShapeSphere(PShape):
    """A class representing a spherical object (3D)."""

    def __init__(self):
        """Initialisation of a shape object describing a sphere (3D)

        Parameter
        ----------
        None
            because a sphere has a predefined shape
        """
        super().__init__(1.0)
        self.aspect_ratio=1.0
        self.ellipse_ratio=1.0
        # can be further extended

    def get_particle_from_size(self, x_ecd, center, rotation=0):
        return SphericalParticle(x_ecd, center)

class PShapeEllipse(PShape):
    def __init__(self, ratio):
        """Initialisation of a shape object describing an ellipse (2D)

        Parameter
        ----------
        ratio : float
            ellipse ratio with values between 0 and 1
        """
        super().__init__(ratio)
        self.aspect_ratio=ratio
        self.ellipse_ratio=ratio
        # can be further extended

    def get_particle_from_size(self, x_ecd, center, rotation=0):
        # get major and minor axis from x_ecd
        lmax=math.sqrt(self.aspect_ratio)*x_ecd
        lmin=self.aspect_ratio*lmax
        return EllipseParticle(lmax, lmin, center, rotation)

class PShapeSuperEllipse(PShape):
    def __init__(self, ratio, exponent):
        """Initialisation of a shape object describing an ellipse (2D)

        Parameter
        ----------
        ratio : float
        ellipse ratio with values between 0 and 1

        """
        super().__init__(ratio)
        self.exponent=exponent
        # TODO: Check if a calculation is possible
        # self.aspect_ratio=ratio
        # self.ellipse_ratio=ratio
        # We can only define the ratio as elongation
        self.elongation=ratio
        # can be further extended

    def get_particle_from_size(self, x_LG, center, rotation=0):
        """Get the particle object from its size, location and rotation

        x_LG : float
            geodesic length, which is used as the particle size
        center : tuple (cx,cy,cz) of float
            geometrical center of the particle in the measurement volume
            This is also the centre of mass for all shapes simulated
            (symmetrical particles).
        rotation : float
            rotation, in radians in range (-PI, PI),
            in contra clockwise direction, with respect to the axis.

        """
        x_E=self.elongation*x_LG
        return SuperEllipseParticle(x_LG, x_E, self.exponent, center, rotation)

class ImageGenerator:
    """A class for generating images with particles drawn"""

    def __init__(self, aShape, FOVWidth=2048, FOVHeight=2048, SVDepth=1000, BackgroundBrightness=255):
        """Initialise the image generating class with static dimensions.
        initialize the Particle generation parameters with defaults as examples.

        Parameters
        ----------
        aShape : PShape object
            A PShape object representing the shape of the particles.
        FOVWidth : integer, optional
            Width of the image frame (Field of View), pixel units
        FOVHeight : integer, optional
            Height of the image frame (Field of View), pixel units
        SVDepth : integer, optional
            Depth of the sampling volume, pixel units
        BackgroundBrightness : integer 0..255, optional
            Brightness of the image background (255 = white, 0 = black)

        Returns
        -------
        Image Generator Object
        """
        self.width, self.height, self.depth = FOVWidth, FOVHeight, SVDepth

        self.background=BackgroundBrightness
        self.foreground=0 # just in case, will be overwritten

        self.pshape = aShape

        """Initialize the particles object lists."""
        # replace with list of the full particle objects for a more detailled
        # subsequent evaluation?
        self.particles_sizes = []
        self.boundary_particles = []

        # Some defaults (overwritten with set_psd())
        # set particle size distribution
        self.r_median, self.sigma_r = 15, 1.3
        self.ismonodisperse=False

    def set_psd(self, x_median, x_stdev=1.3, StrictMonodisperse=False ):
        """Set parameters for the particle size distribution.
        Log-normal distribution of the particle sizes (by number) in pixel units.

        Parameters
        ----------
        x_median : float
            median of the log normal distribution, pixel units
        x_stdev : float, optional, see default
            standard deviation of the log normal distribution, pixel units
        StrictMonodisperse : bool, optional, default is False
            if set to True, stdev_x is ignored and particles are drawn
            monodisperse having the size x_median

        Returns
        -------
        """
        # set particle size distribution
        self.x_median, self.sigma_x = x_median, x_stdev
        self.ismonodisperse=StrictMonodisperse


    def _draw_particles(self, image, ForegroundBrightness):

        for Particle in self.particles:
            Particle.draw(image, ForegroundBrightness)
            if Particle.isdrawn:
                self.particles_sizes.append(Particle.size_x_A)
                self.boundary_particles.append(Particle.isboundary)


    def _place_particle(self, x, image_shape, excludeBorderParticles, MinSpacing):
        """Place the particles inside the generation volume."""
        img_height,img_width = image_shape
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
            r=x/2.0
            # check if the particle touches the image boundaries
            if cx - (r + MinSpacing) <= 0 or cy - (r + MinSpacing) <= 0  or cx + (r + MinSpacing) >= img_width or cy + (r + MinSpacing) >= img_height:
                boundary_particle = True
            else:
                boundary_particle = False

            if (excludeBorderParticles and not boundary_particle) or (not excludeBorderParticles):
                # Check if the particle (as a volume, sphere) overlaps with a particle already in the list.
                if not any(particle.overlap_with(r + MinSpacing, cx, cy, cz)
                           for particle in self.particles):
                    # The particle doesn't overlap any other particle: create it.
                    particle = self.pshape.get_particle_from_size(x, (cx,cy,cz), rotation=(np.random.random()*2.0-1.0)*math.pi)
                    particle.isboundary=boundary_particle # TODO improve code
                    self.particles.append(particle)
                    return
            trials -= 1
        # Warn that the upper limmit of attempts was reached
        print('Warning: limit of trials reached. The current particle is being excluded')

    def _generate_particles(self, image_shape, NumOfParticlesPerImage, excludeBorderParticles, MinSpacing):
        """Generate a new set of particles for each image following the given size distribution"""
        np.random.seed()
        # randomize the number of particles per image
        npp = np.random.poisson(NumOfParticlesPerImage)

        # create a vector containing npp number of size values
        if self.ismonodisperse==False:
            ## Transform normal into log-normal und generate particle size array "x"
            mu, sigma = np.log(self.x_median), np.log(self.sigma_x)
            x = np.exp(np.random.normal(mu, sigma, npp))
        else:
            x =np.full(npp,self.x_median)

        # placing the particles
        self.particles = [] # start with a new list
        for i in range(npp):
            self._place_particle(x[i], image_shape, excludeBorderParticles, MinSpacing)

    def getimage(self, NumOfParticlesPerImage, ForegroundBrightness=0, excludeBorderParticles=False, MinSpacing=1 ):

        # default number of particles per image
        # self.npi = NumOfParticlesPerImage # Is this variable needed
        # self.foreground=ForegroundBrightness # Is this variable needed

        image = np.empty((self.height,self.width), 'uint8')
        image[:,:] = self.background
        # Generate and print the particles image by image
        self._generate_particles(image.shape, NumOfParticlesPerImage, excludeBorderParticles, MinSpacing)
        self._draw_particles(image, ForegroundBrightness)
        return image


##############################################################################
##############################################################################
## Main Code Source
##############################################################################
##############################################################################

# Particles are randomly generated within the sampling volume.
# The particles placed are not allowed to overlap with each other within the volume.

# Drawing parameters
NumOfFrames = 500 # number of frames

# Pixel size and magnification
# In order to compare particle size results calculated by this script in
# micrometers with other software or instruments, provide the correct scaling factors
PixelWidth = 5.5 # unit µm
invMagnification = 1.0
effPixelWidth = invMagnification * PixelWidth

# Histogramm settings
NumberofSizeClasses = 90
x_min = effPixelWidth
x_max = max(2048, 2028)*effPixelWidth

# Create a shape object
# myshape = PShapeSphere()
myshape = PShapeEllipse(0.8)
# myshape = PShapeSuperEllipse(0.1, 20)

# Create the particle drawing object
igenerator = ImageGenerator(myshape, FOVWidth=2048, FOVHeight=2048, SVDepth=2)
# igenerator.set_psd(1.5, 1.0, StrictMonodisperse=True)
igenerator.set_psd(40.0, 2.0, StrictMonodisperse=False)

"""
with TiffWriter('image_stack.tif') as tif:
    for j in range(NumOfFrames):
        image = igenerator.getimage(50)
        print("Frame: ",j + 1)
        tif.write(image, contiguous=True) # cannot be used with compression!
"""

# Generate the particle images
# np.random.seed()
for j in range(NumOfFrames):

    image = igenerator.getimage(400, excludeBorderParticles=False)

    # Collect all images into an image stack
    print("Frame: ",j + 1)
    if(j==0):
        image_stack = image # create the stack from the first image
    else:
        image_stack = np.dstack((image_stack,image))

# Save image stack as TIFF
# The creation of the full image stack in memory limits the number of images.
image_stack = np.transpose(image_stack, (2,0,1))
imsave('image_stack.tif', image_stack, compression='zlib') #, compressionargs={'level': 8}, predictor=True)

print("\neffPixelWidth: ", effPixelWidth)

"""
# Display a single image as an example
from matplotlib import pyplot as plt
plt.imshow(image_stack[1], cmap='gray')
plt.show()
"""

from matplotlib import pyplot as plt
if igenerator.ismonodisperse == False:
    particle_Sizes = np.array(np.asarray(igenerator.particles_sizes), 'float64')*effPixelWidth
    boundary_Particles = igenerator.boundary_particles

    print("\nPSD's p-values of all particles on picture:")
    bins, X, Q_0  = ComputePSD(particle_Sizes, x_min, x_max, NumberofSizeClasses)

    NumOfNonBorderParticles = 0
    for boundaryParticle in igenerator.boundary_particles:
        if not boundaryParticle:
            NumOfNonBorderParticles = NumOfNonBorderParticles+1

    nonborderParticle_Sizes = np.zeros(NumOfNonBorderParticles)

    k=0
    j=0
    for boundaryParticle in igenerator.boundary_particles:
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
