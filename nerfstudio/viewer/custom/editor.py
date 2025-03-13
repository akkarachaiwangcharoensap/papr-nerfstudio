import time
import numpy as np
import viser

from nerfstudio.viewer.scene_object import SceneObject
from nerfstudio.viewer.node import Node


class MeshEditor:
    """
    Main editor class that encapsulates loading and simplifying the mesh,
    creating sphere gizmos, setting up GUI panels and selection/deselection logic,
    and running the main loop.
    """
    def __init__(self, server: viser.ViserServer):
        self._server = server
        self._nodes = []
        self._scene_object = None
        self._tools = []

    def _create_mesh_object(self):
        """
        Add the low-poly mesh to the scene.
        """
        self._scene_object = SceneObject('./dragon.obj')

        self._scene_object_handle = self._server.scene.add_mesh_simple(
            name="/mesh_simple",
            vertices=self._scene_object.get_mesh().vertices,
            faces=self._scene_object.get_mesh().faces,
            wxyz=self._scene_object.get_rotation().wxyz,
            position=tuple(self._scene_object.get_position()),
        )

    def _create_nodes(self):
        scene_object = self._scene_object
        for i, local_pos in enumerate(scene_object.get_mesh().vertices):
            transformed_pos = scene_object.get_rotation().apply(local_pos) + scene_object.get_position()
            new_node = Node(editor=self, index=i, position=transformed_pos)
            self._nodes.append(new_node)

    def get_scene_object(self) -> SceneObject:
        return self._scene_object
    
    def get_scene_object_handle(self) -> viser.MeshHandle:
        return self._scene_object_handle
        
    def get_server(self) -> viser.ViserServer:
        return self._server
    
    def get_nodes(self):
        return self._nodes
    
    def run(self):
        self._create_mesh_object()
        self._create_nodes()