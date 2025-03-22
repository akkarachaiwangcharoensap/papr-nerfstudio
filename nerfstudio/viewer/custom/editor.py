import viser

from nerfstudio.viewer.custom.scene_object import SceneObject
from nerfstudio.viewer.custom.node import Node

class MeshEditor:
    """
    Main editor class that encapsulates loading and rendering the point cloud,
    creating sphere gizmos, setting up GUI panels, and handling selection logic.
    """
    def __init__(self, server: viser.ViserServer):
        self._server = server
        self._nodes = []
        self._scene_object = None
        self._tools = []

        # Load the point cloud from the PLY file.
        self._scene_object = SceneObject('./butterfly_key_points_normed_flipX_pts.ply')

    def _create_nodes(self):
        scene_object = self._scene_object

        # Iterate over the global point positions.
        for i, point in enumerate(scene_object.get_points()):
            new_node = Node(editor=self, index=i, position=point)
            self._nodes.append(new_node)

    def get_scene_object(self):
        return self._scene_object
    
    def get_server(self) -> viser.ViserServer:
        return self._server
    
    def get_nodes(self):
        return self._nodes
    
    def run(self):
        self._create_nodes()