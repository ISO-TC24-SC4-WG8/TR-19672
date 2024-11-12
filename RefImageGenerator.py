# -*- coding: utf-8 -*-
"""
Module for reference image generation for image analysis
"""

import math
import numpy as np

from skimage.draw import disk, ellipse
# from scipy.special import gamma

# from myskidraw import superellipse

def compute_psd(particle_sizes, x_min, x_max, nbins):
    """
    Calculates the cumulative particle size distribution (Q0) and interpolates specific percentiles.

    Parameters
    ----------
    particle_sizes : array_like
        Array containing the sizes of all particles.
    x_min : float
        Minimum size for histogram binning.
    x_max : float
        Maximum size for histogram binning.
    nbins : int
        Number of bins for the histogram.

    Returns
    -------
    bins : np.ndarray
        Array array of the histogram counts. Number of particles per size class.
    X : np.ndarray
        Array of upper interval boundaries of the histogram.
    Q_0 : np.ndarray
        Cumulative distribution by number (fraction of particles).
    interpolated_percentiles : dict
        Dictionary of interpolated values for the percentiles x_10, x_16, x_50, x_84, x_90, x_99.
    """
    if not particle_sizes.size:
        raise ValueError("particle_sizes array is empty.")
    if x_min >= x_max:
        raise ValueError("x_min should be less than x_max.")
    if nbins <= 0:
        raise ValueError("nbins must be a positive integer.")

    # Log-transform particle sizes for log-scale binning
    log_particle_sizes = np.log(particle_sizes)

    # Define equal-width log-scale bin edges
    log_bins = np.histogram_bin_edges(log_particle_sizes, bins=nbins, range=(math.log(x_min), math.log(x_max)))

    # Back-transform to real scale
    bins = np.exp(log_bins)

    # Calculate histogram in real scale
    count, _ = np.histogram(particle_sizes, bins=bins)

    # Cumulative distribution Q0 (as fractions)
    Q_0 = np.cumsum(count) / sum(count)

    # Upper interval boundaries of the histogram
    X = bins[1:]

    # Interpolating values for specific percentiles
    percentiles = [0.10, 0.16, 0.50, 0.84, 0.90, 0.99]
    interpolated_percentiles = {}

    for p in percentiles:
        # Perform log-scale interpolation
        log_interpolated = np.interp(p, Q_0, np.log(X))
        interpolated_value = np.exp(log_interpolated)
        interpolated_percentiles[f"x_{int(p*100)}"] = interpolated_value
        print(f"Q_0: {p:.2f}, X: {interpolated_value:.3f}")

    return bins, X, Q_0, interpolated_percentiles


class Particle:
    """
    A base class representing a particle.
    
    This particle has a geometrical center (cx, cy, cz) and a radius (r) of a minimum circumscribed sphere around this center.
    This information is used to check for mutual overlap within the measurement volume and the overlap with the volume's edge.
    """

    def __init__(self, r_mcc, center):
        """
        Initialize the particle with its geometrical center and radius.
        
        Parameters
        ----------
        r : float
            Radius of the minimum circumscribed sphere around the center. Must be positive.
        center : tuple of float (cx, cy, cz)
            Geometrical center of the particle in the measurement volume.
            This is also the center of mass for all simulated symmetrical particles.
        """
        if not isinstance(r_mcc, (int, float)) or r_mcc <= 0:
            raise ValueError("Radius 'r' must be a positive float.")
        if not (isinstance(center, tuple) and len(center) == 3 and all(isinstance(c, (int, float)) for c in center)):
            raise ValueError("Center must be a tuple of three float values (cx, cy, cz).")

        self.r_mcc = r_mcc
        self.cx, self.cy, self.cz = center
        self.isdrawn = False
        self.isboundary = False

    def _is_within_bounds(self, img_shape):
        """Check if the particle is within the image boundaries."""
        img_height, img_width = img_shape
        return 0 <= self.cx <= img_width and 0 <= self.cy <= img_height
    
    def _touches_boundary(self, img_shape):
        """Check if the particle touches the image boundaries."""
        img_height, img_width = img_shape
        # TODO: Resolve the difference to the boundary check in _place_particle().
        # Define: Where is the coordinate of a pixel? At the center or at the lower left corner of the pixel (preferred)
        # return (
        #     (self.cx - self.r_mcc) < 0 or 
        #     (self.cy - self.r_mcc) < 0 or 
        #     (self.cx + self.r_mcc) > img_width or 
        #     (self.cy + self.r_mcc) > img_height )
        return (
            int(self.cx - self.r_mcc - 0.5) <= 0 or 
            int(self.cy - self.r_mcc - 0.5) <= 0 or 
            int(self.cx + self.r_mcc + 0.5) >= img_width or 
            int(self.cy + self.r_mcc + 0.5) >= img_height )  

    def overlap_with(self, r_mcc, cx, cy, cz):
        """
        Check if the particle overlaps with another particle of radius r_mcc at (cx, cy, cz) in 3D.
        
        This check uses the Euclidean distance and is exact only for spheres. It serves as a simplified check for other shapes.
        TODO: Change logic and names, because overlap cannot be detected exactly for other shapes, but the absence of overlap can be.
        
        Parameters
        ----------
        r : float
            Radius of the minimum circumscribed sphere around the center.
        cx, cy, cz : float
            Geometrical center of the particle position to be checked.
        
        Returns
        -------
        bool
            True if an overlap may occur, False otherwise.
        """
        if not isinstance(r_mcc, (int, float)) or r_mcc <= 0:
            raise ValueError("Radius 'r' must be a positive float.")
        if not all(isinstance(coord, (int, float)) for coord in (cx, cy, cz)):
            raise ValueError("Coordinates (cx, cy, cz) must be float values.")

        # Calculate the Euclidean distance between centers
        # d = math.sqrt((cx - self.cx)**2 + (cy - self.cy)**2 + (cz - self.cz)**2)
        d = np.linalg.norm([cx - self.cx, cy - self.cy, cz - self.cz])

        # Check for overlap based on distance
        return d < (r_mcc + self.r_mcc)

    def __repr__(self):
        return f"Particle(r_mcc={self.r_mcc}, center=({self.cx}, {self.cy}, {self.cz}))"


class CircularParticle(Particle):
    """
    A class representing a circular particle.
    """

    def __init__(self, myshape, x_ecd, center, rotation=0):
        """
        Initialize the spherical particle with its location and equivalent circular diameter.
        Since the shape is defined in myshape, various size parameters can now be set.
        The object carries all size parameters that can be derived from the diameter.
        The sphere does not have an orientation.
        
        Parameters
        ----------
        myshape : class of the particle shape        
        x_ecd : float
            equivalent circulaar diameter.
        center : tuple of float (cx, cy, cz)
            Geometrical center of the particle in the measurement volume.
        """
        # Initialize the parent class with center location and radius used for the minimum circumscribed radius
        super().__init__(x_ecd / 2, center)

        # Definitions from ISO9276-6, size parameters
        # Projection area A       
        self.size_A = myshape.area(x_ecd)
        # Area equivalent diameter x_A
        self.size_x_A = x_ecd
        # Feret Diameters x_Fmax, x_Fmin
        self.size_x_Fmax = x_ecd
        self.size_x_Fmin = x_ecd
        # Ellipse Axes x_Lmax, x_Lmin      
        self.size_x_Lmax = x_ecd
        self.size_x_Lmin = x_ecd
        # Perimeter P
        self.size_P = myshape.perimeter(x_ecd)

    def draw(self, image: np.ndarray, ForegroundBrightness: int, exclude_border):
        """
        Draw the spherical particle on the given 2D image as a disc.
        
        Parameters
        ----------
        image : np.ndarray
            The 2D image on which to draw the particle.
        ForegroundBrightness : int
            The brightness value to use for the particle (e.g., 0-255 for grayscale).
        """
        if not (0 <= ForegroundBrightness <= 255):
            raise ValueError("ForegroundBrightness must be between 0 and 255.")

        # Ensure the image is 2D
        if image.ndim != 2:
            raise ValueError("Image must be a 2D array.")

        # Draw radius
        circle_radius = self.size_x_A / 2
        # Check if the center lies within bounds.     
        self.isboundary = self._touches_boundary(image.shape) 
        self.isdrawn = False
        if not (exclude_border and self.isboundary):
            rr, cc = disk((self.cx, self.cy), circle_radius, shape=image.shape)       
            image[rr, cc] = ForegroundBrightness
            self.isdrawn = True
        
        # if self._is_within_bounds(image.shape):
        #     rr, cc = disk((self.cx, self.cy), circle_radius, shape=image.shape)
        #     # TODO: check whether the following code line makes a difference. It should, because that's what matters.
        #     # it is important not to convert to int. They are explicitly double values
        #     # rr, cc = disk((int(self.cy), int(self.cx)), int(self.r), shape=image.shape)
        #     image[rr, cc] = ForegroundBrightness
        #     self.isdrawn = True

        #     # Check if the particle touches the boundary
        #     self.isboundary = self._touches_boundary(image.shape)
        # else:
        #     self.isdrawn = False
        #     self.isboundary = False

    def __repr__(self):
        return (
            f"SphericalParticle(diameter={self.size_x_A}, "
            f"center=({self.cx}, {self.cy}, {self.cz}), drawn={self.isdrawn}, "
            f"boundary={self.isboundary})"
        )


class EllipseParticle(Particle):
    """
    A class representing an elliptical particle (2D).
    """

    def __init__(self, myshape, x_ecd, center, rotation=0):
        """
        Initialize the ellipse with its location, cx,cy,cz, and axes.
        Since the shape is defined, various size parameters can be set.
        The object carries all size parameters that can be derived from the initial size.

        Keep in mind: This is only a 2D shape with major and minor axis oriented in the image plane and
        not a full 3D ellipsoid.

        Parameters
        ----------
        x_major, x_minor : float
            Major and minor full axes of the ellipse.
        center : tuple (cx, cy, cz) of float
            Geometrical center of the particle in the measurement volume.
        rotation : float
            Rotation of the ellipse in radians (range: -PI to PI), counterclockwise.
        """
        self.x_major = x_ecd / math.sqrt(myshape.ratio)
        self.x_minor = self.x_major * myshape.ratio
        # Initialize the base class with center location and a minimum circumscribed radius
        super().__init__(self.x_major / 2, center)
        self.rotation = rotation

        # Size parameters from ISO9276-6
        # Projection area A
        self.size_A = myshape.area(x_ecd)
        # Area equivalent diameter x_A
        self.size_x_A = x_ecd
        # Feret Diameters x_Fmax, x_Fmin
        self.size_x_Fmax=self.x_major
        self.size_x_Fmin=self.x_minor
        # Ellipse Axes x_Lmax, x_Lmin
        self.size_x_Lmax=self.x_major
        self.size_x_Lmin=self.x_minor
        # Perimeter P - not computed here due to the complexity of exact calculation
        # self.size_P=None

    def _touches_boundary(self, img_shape):
        """Check if the ellipse touches the image boundary."""
        img_height, img_width = img_shape
        half_major = self.size_x_Lmax / 2
        half_minor = self.size_x_Lmin / 2
        # Calculate the extreme points of the rotated ellipse
        max_x = self.cx + half_major * abs(math.cos(self.rotation)) + half_minor * abs(math.sin(self.rotation))
        min_x = self.cx - half_major * abs(math.cos(self.rotation)) - half_minor * abs(math.sin(self.rotation))
        max_y = self.cy + half_major * abs(math.sin(self.rotation)) + half_minor * abs(math.cos(self.rotation))
        min_y = self.cy - half_major * abs(math.sin(self.rotation)) - half_minor * abs(math.cos(self.rotation))

        return ( min_x - 0.5 < 0 or min_y -0.5 < 0 or max_x + 0.5 > img_width or max_y + 0.5 > img_height )

    def draw(self, image, ForegroundBrightness, exclude_border):
        """
        Draw the elliptical particle on the given 2D image.
        
        Parameters
        ----------
        image : np.ndarray
            The image on which to draw the particle.
        ForegroundBrightness : int
            The brightness value to use for the particle.
        """

        if not (0 <= ForegroundBrightness <= 255):
            raise ValueError("ForegroundBrightness must be between 0 and 255.")

        if image.ndim != 2:
            raise ValueError("Image must be a 2D array.")

        # Check if the center lies within bounds.     
        self.isboundary = self._touches_boundary(image.shape) 
        self.isdrawn = False
        if not (exclude_border and self.isboundary):
            rr, cc = ellipse(
                self.cy, self.cx, self.size_x_Lmax / 2, self.size_x_Lmin / 2,
                shape=image.shape, rotation=self.rotation
            )
            image[rr, cc] = ForegroundBrightness            
            self.isdrawn = True

        # # Ensure the particle's center is within the image bounds
        # if self._is_within_bounds(image.shape):
        #     rr, cc = ellipse(
        #         self.cy, self.cx, self.size_x_Lmax / 2, self.size_x_Lmin / 2,
        #         shape=image.shape, rotation=self.rotation
        #     )
        #     image[rr, cc] = ForegroundBrightness
        #     self.isdrawn = True

        #     # Check if the particle touches the image boundary
        #     self.isboundary = self._touches_boundary(image.shape)
        # else:
        #     self.isdrawn = False
        #     self.isboundary = False
            
    def __repr__(self):
        return (
            f"EllipseParticle(major_axis={self.size_x_Lmax}, minor_axis={self.size_x_Lmin}, "
            f"center=({self.cx}, {self.cy}, {self.cz}), rotation={self.rotation}, "
            f"drawn={self.isdrawn}, boundary={self.isboundary})"
        )

# =============================================================================
# class SuperEllipseParticle(Particle):
#     """class representing an superelliptical particle (2D)
#     since the shape is defined, various size parameters can be set.
#     The object carries all size parameters that can be derived from the size
#     given during initialisation (size is given as ecd).
# 
#     TODO: Code needs a revision similar to sphere and ellipse.
#     """
#     def __init__(self, x_major, x_minor, exponent, center, rotation=0):
#         """Initialize the superellipse.
# 
#         Parameter
#         ----------
#         x_major, x_E : float
#             major and minor full axes of the superellipse, here interpreted
#             as geodesic length and thickness
#         exponent : float
#             The overall shape of the superellipse is determined by the value
#             of the exponent
#         center : tuple (cx,cy,cz) of float
#             geometrical center of the particle in the measurement volume
#             This is also the centre of mass for all shapes simulated
#             (symmetrical particles).
#         rotation : float
#             rotation of the ellipse, in radians in range (-PI, PI),
#             in contra clockwise direction, with respect to the axis.
# 
#         Returns
#         ----------
#         the SuperEllipseParticle object
#         """
#         super().__init__(x_major/2.0, center)
#         self.rotation=rotation
#         self.exponent=exponent
#         # Definitions from ISO9276-6, size parameters
#         # Volume V - a 2D shape has no volume
#         # TODO not adapted to a super ellipse!
#         self.size_V=None
#         # Projection area A
#         # The area inside the superellipse can be expressed in terms of
#         # the gamma function
#         self.size_A=x_major*x_minor*math.pow(gamma(1+1/exponent),2)/gamma(1+2/exponent)
#         # Area equivalent diameter x_A
#         self.size_x_A=math.sqrt(4*self.size_A/math.pi)
#         # Feret Diameters x_Fmax, x_Fmin
#         #self.size_x_Fmax=x_major
#         #self.size_x_Fmin=x_minor
#         # Ellipse Axes x_Lmax, x_Lmin
#         #self.size_x_Lmax=x_major
#         #self.size_x_Lmin=x_minor
#         # geodesic length x_LG and thickness x_E
#         self.size_x_LG=x_major
#         self.size_x_E=x_minor
#         # Perimeter P - no simple formula
#         # self.size_P=None
# 
#     def draw(self, image, ForegroundBrightness):
#         img_height,img_width = image.shape
# 
#         self.isdrawn=False
#         #does the particle center lie within the image (dimensions)?
#         if self.cx >= 0 and self.cy >= 0 and self.cx <= img_width and self.cy <= img_height:
#             ### For circular particles ########
#             rr, cc = superellipse(self.cx, self.cy, self.size_x_LG/2, self.size_x_E/2, self.exponent, shape=image.shape, rotation=self.rotation)
#             image[rr, cc] = ForegroundBrightness
#             self.isdrawn=True
# 
#             # self.particles_sizes.append(Particle.r*2)
# 
#             # does the particle touch the the image boundaries (incomplete boundary  particle)?
#             # TODO resolve the difference to the boundary check in _place_particle().
#             # Define: Where is the coordinate of a pixel? at the center or at the lower left corner of the pixel (preferred)
#             if int(self.cx - self.r_mcc + .5) <= 0 or int(self.cy - self.r_mcc + .5) <= 0 or int(self.cx + self.r_mcc +.5) >= img_width or int(self.cy + self.r_mcc + .5) >= img_height:
#                 # self.boundary_particles.append(True)
#                 self.isboundary = True
#             else:
#                 # self.boundary_particles.append(False)
#                 self.isboundary = False
# =============================================================================


class PShape:
    """
    A base class representing a particle shape without assigning a size.
    This is a minimal class that provides a foundation for describing particle shapes
    based on a given proportion descriptor, ratio. This class doesn't yet define specific
    geometric parameters but provides an interface for subclasses to implement more
    specific shape descriptions, such as circles, ellipses, or other geometric forms.

    Attributes
    ----------
    ratio : float
        A proportion descriptor, with values typically between 0 and 1, 
        indicating the shape's aspect ratio or other similar proportion characteristic.
    """

    def __init__(self, ratio):
        """
        Initialize a basic shape with a proportion descriptor.

        Parameters
        ----------
        ratio : float
            Ratio of the proportion descriptor, with values between 0 and 1.
        """
        if not (0 <= ratio <= 1):
            raise ValueError("Ratio must be between 0 and 1.")
        self.ratio=ratio

    def __repr__(self):
        return f"PShape(ratio={self.ratio})"
    
    def radius_mcc(self, x_ecd):
        """
        Placeholder method for calculating the radius of the minimum circumscribed
        circle of the shape which is used for placement checks not just in 2D but
        also in the third dimension of the measurement volume
        
        Returns
        -------
        float
            The radius of the minimum circumscribed circle of the shape. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def area(self, x_ecd):
        """
        Calculate the area based on the equivalent circular diameter.
        Per definition, this is the same for all shapes.
        """
        radius = x_ecd / 2
        return math.pi * radius ** 2

    def perimeter(self, x_ecd):
        """
        Placeholder method for calculating the perimeter.
        
        Returns
        -------
        float
            The perimeter of the shape. Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class PShapeCircle(PShape):
    """A class representing a circle (2D)."""

    def __init__(self):
        """
        Initialize a shape object describing a sphere (3D).

        Since a sphere has a fixed aspect ratio and symmetry, 
        no additional parameters are needed.
        """
        super().__init__(ratio=1.0)
        self.aspect_ratio = 1.0
        self.ellipse_ratio = 1.0
        # can be extended if needed

    def radius_mcc(self, x_ecd):
        """
        Calculate the radius of the minimum circumscribed circle of the shape.
        """
        return x_ecd / 2
    
    def perimeter(self, x_ecd):
        """
        Calculate the perimeter of the based on its diameter.
        """
        return math.pi * x_ecd

    def get_particle_from_size(self, x_ecd, center, rotation=0):
        """
        Generate a SphericalParticle based on the given equivalent circle diameter.

        Parameters
        ----------
        x_ecd : float
            Equivalent circle diameter of the sphere.
        center : tuple of floats
            The center coordinates of the particle in 3D space.
        rotation : float, optional
            Rotation is unused for a sphere, as it has no orientation.
        
        Returns
        -------
        SphericalParticle
            A particle instance with the specified size and position.
        """
        return CircularParticle(self, x_ecd, center)


class PShapeEllipse(PShape):
    """A class representing an elliptical particle shape (2D)."""

    def __init__(self, ratio):
        """
        Initialize a shape object describing an ellipse (2D).

        Parameters
        ----------
        ratio : float
            Aspect ratio of the ellipse, representing the minor-to-major axis ratio.
        """
        super().__init__(ratio)
        self.aspect_ratio=ratio
        self.ellipse_ratio=ratio
        # can be further extended

    def radius_mcc(self, x_ecd):
        """
        Calculate the radius of the minimum circumscribed circle of the ellipse
        which is x_major / 2
        
        Returns
        -------
        float
            The radius of the minimum circumscribed circle of the shape.
        """
        return x_ecd / math.sqrt(self.ellipse_ratio)

    # def area(self, x_major, x_minor):
    #     """Calculate the area of the ellipse based on its major and minor axes."""
    #     return math.pi * (x_major / 2) * (x_minor / 2)
   
    def perimeter(self, x_major, x_minor):
        """
        Approximate the perimeter of the ellipse using Ramanujan's approximation.
        
        Parameters
        ----------
        x_major : float
            Length of the major axis.
        x_minor : float
            Length of the minor axis.
        
        Returns
        -------
        float
            Approximate perimeter of the ellipse.
        """
        a, b = x_major / 2, x_minor / 2
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))

    def get_particle_from_size(self, x_ecd, center, rotation=0):
        """
        Generate an EllipseParticle based on the equivalent circle diameter.

        Parameters
        ----------
        x_ecd : float
            Equivalent circle diameter for an ellipse with the same area.
        center : tuple of floats
            The center coordinates of the particle in the 2D space.
        rotation : float
            The rotation angle of the ellipse, in radians.
        
        Returns
        -------
        EllipseParticle
            A particle instance with the specified size, position, and orientation.
        """
        return EllipseParticle(self, x_ecd, center, rotation)

# =============================================================================
# class PShapeSuperEllipse(PShape):
#     def __init__(self, ratio, exponent):
#         """Initialisation of a shape object describing an ellipse (2D)
#         TODO: Code needs a revision similar to sphere and ellipse.
# 
#         Parameter
#         ----------
#         ratio : float
#         ellipse ratio with values between 0 and 1
# 
#         """
#         super().__init__(ratio)
#         self.exponent=exponent
#         # TODO: Check if a calculation is possible
#         # self.aspect_ratio=ratio
#         # self.ellipse_ratio=ratio
#         # We can only define the ratio as elongation
#         self.elongation=ratio
#         # can be further extended
# 
#     def get_particle_from_size(self, x_LG, center, rotation=0):
#         """Get the particle object from its size, location and rotation
# 
#         x_LG : float
#             geodesic length, which is used as the particle size
#         center : tuple (cx,cy,cz) of float
#             geometrical center of the particle in the measurement volume
#             This is also the centre of mass for all shapes simulated
#             (symmetrical particles).
#         rotation : float
#             rotation, in radians in range (-PI, PI),
#             in contra clockwise direction, with respect to the axis.
# 
#         """
#         x_E=self.elongation*x_LG
#         return SuperEllipseParticle(x_LG, x_E, self.exponent, center, rotation)
# =============================================================================

class ImageGenerator:
    """A class for generating images with particles drawn"""

    def __init__(self, aShape, FOVWidth=2048, FOVHeight=2048, SVDepth=1000, BackgroundBrightness=255):
        """
        Initialize the image generator with frame dimensions and particle shape.

        Parameters
        ----------
        aShape : PShape object
            Particle shape object, defining particle type and aspect ratio.
        FOVWidth : int, optional
            Image frame width (Field of View), in pixels.
        FOVHeight : int, optional
            Image frame height (Field of View), in pixels.
        SVDepth : int, optional
            Depth of the sampling volume, in pixels.
        BackgroundBrightness : int, optional
            Brightness of the background (0=black, 255=white).
        """
        self.width, self.height, self.depth = FOVWidth, FOVHeight, SVDepth
        self.background_brightness = BackgroundBrightness
        self.pshape = aShape

        # Defaults for particle size distribution
        self.x_median, self.sigma_x = 15, 1.3
        self.ismonodisperse = False

        # Cumulative data for all generated particles (cummulative over multiple images)
        # This could be replaced by a list of the full particle objects for later evaluation or verification
        self.particles_sizes = []
        self.boundary_particles = []               

    def set_psd(self, x_median, x_stdev=1.3, strict_monodisperse=False ):
        """
        Set parameters for the particle size distribution.
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
        """
        # set particle size distribution
        self.x_median, self.sigma_x = x_median, x_stdev
        self.ismonodisperse=strict_monodisperse

    def reset_particles_data(self):
        """Reset cumulative data for particles across all prior generated images."""
        self.particles_sizes.clear()
        self.boundary_particles.clear()

    def _draw_particles(self, image, foreground_brightness, exclude_border):
        """Draw all particles on the image and store their data."""
        for particle in self.particles:
            particle.draw(image, foreground_brightness, exclude_border)
            # if particle.isdrawn:
            self.particles_sizes.append(particle.size_x_A)
            self.boundary_particles.append(particle.isboundary)

    def _place_particle(self, size, min_spacing):
        """
        Attempt to place a particle in the image without overlap.
        
        Parameters
        ----------
        size : float
            Diameter of the particle.
        exclude_border : bool
            Whether to exclude particles touching image borders.
        min_spacing : float
            Minimum spacing required between particles.
        """
        # The trials number: if a particle with a given size is not succesfully placed within this number
        # of trials, the placement process is stopped .
        trials = 100
        while trials:
            
            # Create a location and the minimum circumscribed radius of the shape
            # - particles having their center outside the volume are not generated even
            #   if they extend beyond the border of the picture into the picture.
            # - the generation volume is the same vor all particles
            # - all particles placed are included in our list of particles
            # - if the particle is actually drawn depends on the call to the drawing procedure          
            cx, cy = np.random.uniform(0, self.width), np.random.uniform(0, self.height)
            cz = np.random.uniform(0, self.depth)
            r_mcc = self.pshape.radius_mcc(size)
            
            # check if this location overlapps with already placed particle in our list
            # if not, create the particle object
            if not any(particle.overlap_with(r_mcc + min_spacing, cx, cy, cz) for particle in self.particles):
                new_particle = self.pshape.get_particle_from_size(size, (cx, cy, cz), rotation=np.random.uniform(-math.pi, math.pi))
                self.particles.append(new_particle)
                return
            trials -= 1

        # Warn that the upper limmit of attempts was reached
        print('Warning: limit of trials reached. Particle excluded.')

    def _generate_particle_sizes(self, num_particles):
        """Generate particles based on the set size distribution."""
        # randomize the number of particles per image
        np.random.seed()
        num_particles = np.random.poisson(num_particles)

        # Generate particle sizes
        if self.ismonodisperse:
            sizes = np.full(num_particles, self.x_median)
        else:
            mu, sigma = np.log(self.x_median), np.log(self.sigma_x)
            sizes = np.exp(np.random.normal(mu, sigma, num_particles))
        return sizes


    def getimage(self, num_particles, foreground_brightness=0, exclude_border=False, min_spacing=1 ):
        """
        Generate an image with particles drawn according to specified parameters.

        Parameters
        ----------
        num_particles : int
            Expected number of particles in the image.
        foreground_brightness : int, optional
            Brightness value of the particles (0=black, 255=white).
        exclude_border : bool, optional
            Whether to exclude particles touching the image borders.
        min_spacing : float, optional
            Minimum spacing between particles.
        
        Returns
        -------
        np.ndarray
            Generated image with particles.
        """
        image = np.full((self.height, self.width), self.background_brightness, dtype=np.uint8)

        # Generate and print the particles image by image
        sizes = self._generate_particle_sizes(num_particles)
        self.particles = [] #  re-init list of particles to be drawn
        for size in sizes:
            self._place_particle(size, min_spacing)
        self._draw_particles(image, foreground_brightness, exclude_border)
        
        return image


##############################################################################
## Definition of constants used for the example main
##############################################################################

# Drawing parameters
NUM_OF_FRAMES = 200 # number of frames
WIDTH = 2048        # image resolution, dimension 1
HEIGHT = 2048       # image resolution, dimension 2
SVDEPTH = 2         # depth, dimension 3 of measurement volume

# Pixel size and magnification
# In order to compare particle size results calculated by this script in micrometers
# with other software or instruments, provide the correct scaling factors
PIXEL_WIDTH = 5.5  # unit µm
INV_MAGNIFICATION = 1.0
EFF_PIXEL_WIDTH = INV_MAGNIFICATION * PIXEL_WIDTH

# Histogramm settings for the particle size distribution
NUM_OF_SIZE_CLASSES = 90
X_MIN = EFF_PIXEL_WIDTH
X_MAX = max(WIDTH, HEIGHT)*EFF_PIXEL_WIDTH

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tifffile import imsave #, TiffWriter
    # Example usage demonstrating the functionality of the classes
    # Instantiate a shape object, for example an ellipse
    # my_shape = PShapeCircle()
    my_shape = PShapeEllipse(ratio=0.75)  
    # my_shape = PShapeSuperEllipse(0.1, 20)

    # Instantiate the ImageGenerator with the ellipse shape
    image_generator = ImageGenerator(
        aShape=my_shape,
        FOVWidth=WIDTH,
        FOVHeight=HEIGHT,
        SVDepth=SVDEPTH,
        BackgroundBrightness=255
    )

    # Set the particle size distribution for demonstration
    image_generator.set_psd(
        x_median=60,       # Median particle size (equivalent diameter)
        x_stdev=1.8,       # Standard deviation for log-normal distribution
        strict_monodisperse=False
    )

    # Generate the particle images
    for j in range(NUM_OF_FRAMES):
        image = image_generator.getimage(
            num_particles=50,         # Average number of particles per image
            foreground_brightness=0,  # Particles will be black
            exclude_border=True,     # Exclude particles touching image borders?
            min_spacing=2             # Minimum spacing between particles
        )
        # Collect all images into an image stack
        print("Frame: ",j + 1)
        if(j==0):
            image_stack = image # create the stack from the first image
        else:
            image_stack = np.dstack((image_stack,image))

    # Save image stack as TIFF multiple page images
    # The creation of the full image stack in memory limits the number of images.
    image_stack = np.transpose(image_stack, (2,0,1))
    imsave('image_stack.tif', image_stack, compression='zlib') #, compressionargs={'level': 8}, predictor=True)
    
    # Unfortunately wrinting the frames subsequently on disk with the following code does not support compression:
    # with TiffWriter('image_stack.tif') as tif:
    #     for j in range(NumOfFrames):
    #         image = igenerator.getimage(50)
    #         print("Frame: ",j + 1)
    #         tif.write(image, contiguous=True) # cannot be used with compression!


    print("\neffPixelWidth: ", EFF_PIXEL_WIDTH)

    """
    # Display a single image as an example
    from matplotlib import pyplot as plt
    plt.imshow(sample_image, cmap='gray')
    plt.title("Generated Image with Particles")
    plt.axis('off')
    plt.show()
    """

    # For the polydisperse case, calculate a cumulative particle size distribution from the particle size data
    # and output the persentiles of the distribution 
    if not image_generator.ismonodisperse:

        # Convert particle sizes from pixels to µm
        particle_sizes = np.array(image_generator.particles_sizes) * EFF_PIXEL_WIDTH
        boundary_flags = np.array(image_generator.boundary_particles, dtype=bool)
        # Get sizes of non-boundary particles using boolean masking
        nonborder_particle_sizes = particle_sizes[~boundary_flags]

        print("\nNumber of particles: ", len(particle_sizes))
        print("\nNumber of non border particles: ", len(nonborder_particle_sizes))

        print("\nPSD's p-values of all particles")
        bins, X, Q_0, _  = compute_psd(particle_sizes, X_MIN, X_MAX, NUM_OF_SIZE_CLASSES)
        print("\nPSD's p-values of all non-border particles:")
        bins_non, X_non, Q_0_non, gpercentiles  = compute_psd(nonborder_particle_sizes, X_MIN, X_MAX, NUM_OF_SIZE_CLASSES)

        plt.semilogx(X, Q_0, label="All Particles")
        if X_non is not None and Q_0_non is not None:
            plt.semilogx(X_non, Q_0_non, label="Non-Border Particles")
        plt.xlabel("Particle Size (µm)")
        plt.ylabel("Cumulative Distribution (%)")
        plt.legend()
        plt.show()
    
