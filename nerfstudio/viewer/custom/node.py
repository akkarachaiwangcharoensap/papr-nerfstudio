import numpy as np

class Node:
    """
    Represents a single sphere (vertex) with an attached transform control.
    Registers update and drag callbacks so that when the control is moved, the
    corresponding mesh vertex is updated.
    """
    def __init__(self, editor, index: int, position: np.ndarray):
        self._editor = editor
        self._index = index
        self._server = editor.get_server()

        self._scene_object = self._editor.get_scene_object()
        self._wxyz = self._scene_object.get_rotation().wxyz

        self._selected = False
        self._default_color = [0, 0, 255]

        # Add a blue sphere to the scene.
        self._object = self._server.scene.add_point_cloud(
            name=f"/sphere_blue_{self._index}",
            point_size=0.3,
            point_shape='rounded',
            colors=self._default_color,
            position=tuple(position),
            points=np.atleast_2d([0, 0, 0]),
            wxyz=self._wxyz,
            visible=True,
        )

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

        # Register callbacks for this sphere's transform control.
        self._register_events()

    def _register_events(self):
        self._control_handle.on_update(self._on_move)

    def _on_move(self):
        """
        Called every time the transform control moves the sphere.
        Updates the sphere's position and the associated low-poly mesh vertex.
        """
        new_position = np.array(self._control_handle.position)
        self._object.position = tuple(new_position)
        print(f"Sphere /sphere_blue_{self._index} moved to position: {new_position}")

        # Find the closest vertex (by global position).
        distances = np.linalg.norm(self._editor.global_vertices - new_position, axis=1)
        min_index = np.argmin(distances)
        closest_global = self._editor.global_vertices[min_index]
        print(f"Closest vertex index {min_index} with global position: {closest_global}")

        # Compute the new vertex position in the mesh's local frame.
        position = self._scene_object.get_position()
        rotation = self._scene_object.rotation()
        mesh = self._scene_object.get_mesh()
        global_vertices = self._scene_object.get_global_vertices()

        # Update position.
        new_local = rotation.inverse().apply(new_position - position)
        mesh.vertices[min_index] = new_local
        global_vertices[min_index] = new_position

        # Update the mesh in the scene.
        scene_object_handle = self._editor.get_scene_object_handle()
        scene_object_handle.vertices = self._editor.low_poly_mesh.vertices
        print(f"Updated vertex {min_index} to new local position: {new_local}")

    def on_selected(self):
        self._selected = True
        # Remove the existing point cloud from the scene using the remove() function.
        self._object.remove()
        # Recreate the point cloud with the new selected color (green).
        self._object = self._server.scene.add_point_cloud(
            name=f"/sphere_green_{self._index}",
            point_size=0.3,
            point_shape='rounded',
            colors=[0, 255, 0],  # selected color: green
            position=tuple(self.get_position()),
            points=np.atleast_2d([0, 0, 0]),
            wxyz=self._wxyz,
            visible=True,
        )
        # Update the transform control handle if needed.
        self._control_handle.position = tuple(self.get_position())
        self._control_handle.wxyz = self._wxyz

    def on_deselected(self):
        self._selected = False
        # Remove the current point cloud using the remove() method.
        self._object.remove()
        # Recreate the point cloud with the default color (blue).
        self._object = self._server.scene.add_point_cloud(
            name=f"/sphere_blue_{self._index}",
            point_size=0.3,
            point_shape='rounded',
            colors=self._default_color,  # default color: e.g., [0, 0, 255]
            position=tuple(self.get_position()),
            points=np.atleast_2d([0, 0, 0]),
            wxyz=self._wxyz,
            visible=True,
        )
        # Update the control handle as needed.
        self._control_handle.position = tuple(self.get_position())
        self._control_handle.wxyz = self._wxyz

    def get_object(self):
        return self._object
    
    def get_control_handle(self):
        return self._control_handle

    def get_position(self):
        return self._object.position