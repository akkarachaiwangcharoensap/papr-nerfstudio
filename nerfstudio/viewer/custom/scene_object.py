import numpy as np
import trimesh
import viser

class SceneObject:
    def __init__(self, file_path):
        # Load the PLY file as a point cloud.
        self._point_cloud = trimesh.load(file_path, process=False)
        
        # Check if vertices exist (i.e. the file contains point cloud vertices).
        if hasattr(self._point_cloud, 'vertices'):
            self._points = self._point_cloud.vertices
        else:
            raise ValueError("Loaded file does not contain point cloud vertices.")
        
        # Define a transformation: rotate 90° about the X-axis and no translation.
        self._rotation = viser.transforms.SO3.from_x_radians(np.pi / 2)
        self._position = np.array([0.0, 0.0, 0.0])
        
        # Compute global positions of the points.
        self._global_points = self._rotation.apply(self._points) + self._position
    
    def get_points(self) -> np.ndarray:
        return self._global_points
    
    def get_position(self) -> np.ndarray:
        return self._position
    
    def get_rotation(self):
        return self._rotation