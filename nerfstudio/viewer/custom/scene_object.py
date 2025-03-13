import numpy as np
import trimesh
import viser

class SceneObject:
    def __init__(self, file_path):
        # Load and scale the original mesh.
        self._mesh = trimesh.load_mesh(file_path)
        assert isinstance(self._mesh, trimesh.Trimesh), "Mesh failed to load as a Trimesh"

        self._mesh.apply_scale(1)

        # Simplify the mesh to create a low-poly version.
        self._low_poly_mesh = self._mesh.simplify_quadric_decimation(1)

        # Define mesh transformation: rotation and translation.
        self._mesh_rot = viser.transforms.SO3.from_x_radians(np.pi / 2)
        self._mesh_pos = np.array([0.0, 0.0, 0.0])

        # Precompute the global positions of the low-poly mesh vertices.
        self._global_vertices = self._mesh_rot.apply(self._low_poly_mesh.vertices) + self._mesh_pos
    
    def get_mesh(self) -> trimesh.Trimesh:
        return self._low_poly_mesh
    
    def get_global_vertices(self) -> np.ndarray:
        return self._global_vertices
    
    def get_position(self) -> np.array:
        return self._mesh_pos
    
    def get_rotation(self) -> viser.transforms.SO3:
        return self._mesh_rot