import numpy as np

class Node:
    """
    Represents a single sphere (vertex) with an attached transform control.
    Instead of removing/adding point clouds on selection changes, we create two
    copies: one for the deselected (blue) state and one for the selected (green) state.
    We then toggle the visibility of these objects.
    """
    def __init__(self, editor, index: int, position: np.ndarray):
        self._editor = editor
        self._index = index
        self._server = editor.get_server()

        self._size = 0.01
        self._default_color = [0, 0, 255]      # Deselected color: blue
        self._selected_color = [0, 255, 0]       # Selected color: green

        self._scene_object = self._editor.get_scene_object()
        self._wxyz = self._scene_object.get_rotation().wxyz

        self._selected = False

        # Create the deselected (blue) point cloud and set it as initially visible.
        self._object_deselected = self._server.scene.add_point_cloud(
            name=f"/sphere_blue_{self._index}",
            point_size=self._size,
            point_shape='rounded',
            colors=self._default_color,
            position=tuple(position),
            points=np.atleast_2d([0, 0, 0]),
            wxyz=self._wxyz,
            visible=True,
        )
        # Create the selected (green) point cloud and set it as initially hidden.
        self._object_selected = self._server.scene.add_point_cloud(
            name=f"/sphere_green_{self._index}",
            point_size=self._size,
            point_shape='rounded',
            colors=self._selected_color,
            position=tuple(position),
            points=np.atleast_2d([0, 0, 0]),
            wxyz=self._wxyz,
            visible=False,
        )
        # Keep a reference to the currently visible object.
        self._object = self._object_deselected

        # Add a transform control (gizmo) for the sphere.
        self._control_handle = self._server.scene.add_transform_controls(
            name=f"/transform/sphere_blue_{self._index}",
            opacity=1,
            scale=1,
            disable_rotations=True,
            visible=False,
        )
        self._control_handle.position = tuple(position)
        self._control_handle.wxyz = self._wxyz

        # Register callbacks for transform updates.
        self._register_events()

    def _register_events(self):
        self._control_handle.on_update(self._on_move)

    def _on_move(self):
        """
        Called whenever the transform control moves the sphere.
        Updates both point cloud objects' positions and handles the mesh vertex update.
        """
        new_position = np.array(self._control_handle.position)
        # Update positions of both point clouds so they remain in sync.
        self._object_deselected.position = tuple(new_position)
        self._object_selected.position = tuple(new_position)
        print(f"Sphere /sphere_blue_{self._index} moved to position: {new_position}")

        # Example: Update the associated mesh vertex (existing logic).
        distances = np.linalg.norm(self._editor.global_vertices - new_position, axis=1)
        min_index = np.argmin(distances)
        closest_global = self._editor.global_vertices[min_index]
        print(f"Closest vertex index {min_index} with global position: {closest_global}")

        position = self._scene_object.get_position()
        rotation = self._scene_object.rotation()
        mesh = self._scene_object.get_mesh()
        global_vertices = self._scene_object.get_global_vertices()

        # Compute the new vertex position in the mesh's local frame.
        new_local = rotation.inverse().apply(new_position - position)
        mesh.vertices[min_index] = new_local
        global_vertices[min_index] = new_position

        # Update the mesh in the scene.
        scene_object_handle = self._editor.get_scene_object_handle()
        scene_object_handle.vertices = self._editor.low_poly_mesh.vertices
        print(f"Updated vertex {min_index} to new local position: {new_local}")

    def on_selected(self):
        """
        Called when the node is selected. Instead of recreating the point cloud,
        we hide the deselected object and show the selected one.
        """
        sync_pos = self.get_position()
        self._selected = True

        self._object_deselected.visible = False
        self._object_selected.visible = True

        self._object = self._object_selected
        self._object.position = tuple(sync_pos)

        # Update transform control handle if needed.
        self._control_handle.position = tuple(self.get_position())
        self._control_handle.wxyz = self._wxyz

    def on_deselected(self):
        """
        Called when the node is deselected. Toggles visibility so that the deselected
        (blue) point cloud is shown.
        """
        sync_pos = self.get_position()
        self._selected = False

        self._object_selected.visible = False
        self._object_deselected.visible = True
        
        self._object = self._object_deselected
        self._object.position = tuple(sync_pos)

        # Update transform control handle if needed.
        self._control_handle.position = tuple(self.get_position())
        self._control_handle.wxyz = self._wxyz

    def get_position(self):
        """
        Returns the position of the currently visible point cloud.
        Assumes that both point clouds are kept in sync.
        """
        return np.array(self._object.position)