"""
Structure from Motion bundle adjusment module.

Based on: https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

Authors: Caio Cancian, Felipe Moran

Version: 1.0.0

"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

class BundleAdjuster:
    """Bundle adjusment base class.

    Performs end-to-end bundle adjusment to Structure from Motion problems.
    """

    def __init__(self, camera_matrix, tol=1e-4, method='trf', verbose=0):

        # Basic optimization attributes
        self.tol = tol
        self.method = method
        self.verbose = verbose
        self.camera_matrix = camera_matrix

        # Attributes to store resulting optimization parameters
        self.optimized_points = None
        self.optimized_cameras = None

    def _rotate(self, points, rot_vecs):
        """Rotate 3D points by given rotation vectors.

        Rodrigues' rotation formula is used.
        See https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        for details
        """
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rot_vecs / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rot = cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v
        return rot

    def _project(self, points, camera_params):
        """Convert 3-D points to 2-D by projecting onto images.

        No radial distortion is considered.
        """

        # Rotate and translate
        points_proj = self._rotate(points, camera_params[:, :3])
        points_proj += camera_params[:, 3:6]

        # Divide by scale
        points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]

        # Set camera calibration params
        fx = self.camera_matrix[0,0]
        cx = self.camera_matrix[0,2]
        fy = self.camera_matrix[1,1]
        cy = self.camera_matrix[1,2]

        # Apply projection formula WITHOUT radial distortion
        points_proj[:,0] = points_proj[:,0]*fx + cx
        points_proj[:,1] = points_proj[:,1]*fy + cy

        return points_proj

    def _objective_function(self, params, n_cameras, n_points, camera_indices,
                            point_indices, points_2d):
        """Compute residuals.

        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = self._project(points_3d[point_indices],
                                    camera_params[camera_indices])
        return (points_proj - points_2d).ravel()

    def _bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices,
                                    point_indices):
        """Build optimization sparse matrix."""

        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

        return A

    def _get_optimized_params(self, params, n_cameras, n_points):
        """ Parse optimization results to camera params and 3D points"""
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        return camera_params, points_3d

    def optimize(self,camera_params, points_3d, points_2d, camera_indices,
                points_indices):
        """Apply bundle adjustment optimization

        Parameters
        ----------
        camera_params: array, shape (n_cameras, 6)
            Contains initial camera parameter estimates. For each row,
            the parameters must be organized as 3 rotations, then 3
            translations in the camera reference frame.

        points_3d: array, shape (n_points, 3)
            Contains inital 3D points coordinates estimates in the same
            reference frame.

        points_2d: array, shape (n_observations, 2)
            Contains the 2D coordinates of every observed 3D point projected
            using one of the cameras.

        camera_indices: array, shape (n_observations,)
            Contains camera indices for each observed projection. The i-th
            element of this array corresponds to the camera that generated
            the i-th 2D point.

        points_indices: array, shape (n_observations,)
            Contains 3D points indices for each observed projection. The i-th
            element of this array corresponds to the 3D point that generated
            the i-th 2D point.

        Returns
        -------
        optimized_cameras: array, shape (n_cameras, 9)
            Contains optimized camera parameters. Same order as input.

        optimized_points: array, shape (n_points, 3)
            Contains optimized 3D points coordinates.
        """

        # Get parameters of interest
        n_cameras = camera_params.shape[0]
        n_points = points_3d.shape[0]

        # Initilize optimization with
        x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

        # Build sparse matrix and run optimization
        A = self._bundle_adjustment_sparsity(n_cameras, n_points,
                                            camera_indices, point_indices)
        optim_res = least_squares(self._objective_function, x0,
                                  jac_sparsity=A,
                                  verbose=self.verbose,
                                  x_scale='jac',
                                  ftol=self.tol,
                                  method=self.method,
                                  args=(n_cameras, n_points, camera_indices,
                                        point_indices, points_2d))

        # Return optimized params
        self.optimized_cameras, self.optimized_points = self._get_optimized_params(optim_res.x, n_cameras, n_points)

        return self.optimized_cameras, self.optimized_points
