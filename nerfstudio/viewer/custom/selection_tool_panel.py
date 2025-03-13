# from __future__ import annotations

# from pathlib import Path

# import viser
# import viser.transforms as vtf
# from typing_extensions import Literal

# from nerfstudio.data.scene_box import OrientedBox
# from nerfstudio.models.base_model import Model
# from nerfstudio.models.splatfacto import SplatfactoModel
# from nerfstudio.viewer.viewer_elements import ViewerRectSelect

# from nerfstudio.viewer.viewer_elements import ( 
#     ViewerRectSelect,
#     ViewerControl
# )

# def populate_selection_tool_tab(
#     server: viser.ViserServer,
#     config_path: Path,
#     viewer_model: Model,
# ) -> None:
        
#     # Listen to rectangle selections in the viewer...
#     def pointer_rect_cb(rect: ViewerRectSelect):
#         print(f"Rectangular selection from {rect.min_bounds} to {rect.max_bounds}.")
    
#     _select_button = server.gui.add_button("Select Points")
#     _select_button.on_click(pointer_rect_cb)

from pathlib import Path
import viser
from nerfstudio.models.base_model import Model
from nerfstudio.viewer.viewer_elements import ViewerRectSelect
from nerfstudio.viewer.node import Node

from nerfstudio.viewer.editor import MeshEditor

import trimesh
import numpy as np

class SelectionToolTab:
    def __init__(self, server: viser.ViserServer, config_path: Path, viewer_model: Model):
        self._server = server
        self.config_path = config_path
        self.viewer_model = viewer_model

        self._editor = MeshEditor(server)
        self._editor.run()
        
        self._scene_object = self._editor.get_scene_object()

        # Dictionary to track the current selection state.
        self._current_selection = {
            "indices": None, 
            "group_gizmo": None, 
            "initial_center": None
        }
        
        self._initialize_ui()

    def _initialize_ui(self):
        # UI Elements
        self._select_button = self._server.gui.add_button("Select Points")
        self._deselect_button =  self._server.gui.add_button("Deselect Points")
        self._deselect_button.disabled = True  # Initially, no selection exists.

        # Register button click callbacks.
        self._select_button.on_click(self._select_click)
        self._deselect_button.on_click(self._deselect_click)

    def _on_rect_selection(self, event: viser.ScenePointerEvent):
        """
        Handle rectangular selection events by projecting node positions into
        screen space, filtering out occluded nodes, and creating a group gizmo.
        
        Parameters:
            event (viser.ScenePointerEvent): Contains pointer, screen, and camera data.
        """
        # Reset previous selection state.
        self._reset_selection_state(event)

        camera = event.client.camera

        # Get node positions from the editor.
        nodes_positions = self._get_nodes_positions()

        # Transform the positions to the camera coordinate frame.
        sphere_camera = self._transform_to_camera_frame(nodes_positions, camera)

        # Project the 3D points to 2D screen space.
        proj = self._project_to_screen(sphere_camera, camera)

        # Determine the rectangular selection bounds.
        rect_min, rect_max = self._get_rectangle_bounds(event.screen_pos)

        # Find candidate nodes whose projected positions lie in the rectangle.
        candidate_indices = self._get_vertices_in_rect(proj, rect_min, rect_max)

        # Filter candidates by checking for occlusion.
        visible_indices = self._filter_visible_indices(candidate_indices, nodes_positions, camera)

        if not visible_indices:
            print("No visible points selected")
            return

        print(f"Selected visible points: {visible_indices}")
        
        # Find the center point to position group gizmo.
        selected_positions = nodes_positions[visible_indices]
        group_center = selected_positions.mean(axis=0)

        # Create a group gizmo at the center of the selected nodes.
        group_gizmo, group_initial_center = self._create_group_gizmo(group_center, len(visible_indices))

        # Change the color of selected nodes to green.
        self._select_nodes(visible_indices)

        # Save the selection state and enable the deselect button.
        self._update_selection_state(visible_indices, group_gizmo, group_initial_center)

        # Register an update callback for the group gizmo.
        group_gizmo.on_update(
            lambda event: self._group_selection(group_gizmo, visible_indices, group_initial_center)
        )

    def _reset_selection_state(self, event: viser.ScenePointerEvent):
        """
        Reset the selection state by deselecting nodes, removing the pointer callback,
        and enabling the select button.
        """
        self._deselect_nodes()
        event.client.scene.remove_pointer_callback()
        self._select_button.disabled = False

    def _get_nodes_positions(self) -> np.ndarray:
        """
        Retrieve the current positions of all nodes from the editor.
        """
        return np.array([np.array(node.get_position()) for node in self._editor.get_nodes()])
    
    def _transform_to_camera_frame(self, nodes_positions: np.ndarray, camera) -> np.ndarray:
        """
        Transform node positions from world coordinates to the camera coordinate frame.
        """
        rows = nodes_positions.shape[0]
        ones = np.ones((rows, 1))
        node_hom = np.hstack([nodes_positions, ones])
        R_camera_world = viser.transforms.SE3.from_rotation_and_translation(
            viser.transforms.SO3(camera.wxyz), camera.position
        ).inverse()
        node_camera = (R_camera_world.as_matrix() @ node_hom.T).T[:, :3]
        return node_camera

    def _project_to_screen(self, node_camera: np.ndarray, camera) -> np.ndarray:
        """
        Project 3D points from the camera frame into 2D screen space.
        """
        fov, aspect = camera.fov, camera.aspect
        proj = node_camera[:, :2] / node_camera[:, 2:3]
        proj /= np.tan(fov / 2)
        proj[:, 0] /= aspect
        proj = (1 + proj) / 2
        return proj

    def _get_rectangle_bounds(self, screen_pos: tuple):
        """
        Compute the min and max corners of the selection rectangle.
        """
        rect_min = np.minimum(screen_pos[0], screen_pos[1])
        rect_max = np.maximum(screen_pos[0], screen_pos[1])
        return rect_min, rect_max

    def _get_vertices_in_rect(
        self, proj: np.ndarray, rect_min: np.ndarray, rect_max: np.ndarray
    ) -> np.ndarray:
        """
        Determine which projected points lie within the selection rectangle.
        """
        vertices = np.where(
            (proj[:, 0] >= rect_min[0])
            & (proj[:, 0] <= rect_max[0])
            & (proj[:, 1] >= rect_min[1])
            & (proj[:, 1] <= rect_max[1])
        )[0]
        return vertices

    def _is_visible(self, node_pos: np.ndarray, camera) -> bool:
        """
        Check if a node at a given world position is visible (i.e. not occluded) from the camera.
        """
        ray_origin = np.array(camera.position)
        ray_dir = node_pos - ray_origin
        distance_to_node = np.linalg.norm(ray_dir)
        if distance_to_node < 1e-6:
            return True

        ray_dir_norm = ray_dir / distance_to_node

        # Transform ray to local coordinates of the scene object.
        ray_origin_local = self._scene_object.get_rotation().inverse().apply(
            ray_origin - self._scene_object.get_position()
        )
        ray_dir_local = self._scene_object.get_rotation().inverse().apply(ray_dir_norm)

        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(
            self._scene_object.get_mesh()
        )
        hit_positions, _, _ = intersector.intersects_location(
            ray_origin_local.reshape(1, 3),
            ray_dir_local.reshape(1, 3),
            multiple_hits=False,
        )

        if len(hit_positions) > 0:
            hit_global = (
                self._scene_object.get_rotation().apply(hit_positions[0])
                + self._scene_object.get_position()
            )
            hit_distance = np.linalg.norm(hit_global - ray_origin)
            # If an intersection is detected before reaching the node, it is occluded.
            if hit_distance < distance_to_node - 0.01:
                return False
        return True

    def _filter_visible_indices(
        self, candidate_indices: np.ndarray, nodes_positions: np.ndarray, camera
    ) -> list:
        """
        Filter candidate node indices to include only those that are visible from the camera.
        
        Parameters:
            candidate_indices (np.ndarray): Array of candidate indices based on the rectangle.
            nodes_positions (np.ndarray): Array of all node positions.
            camera: Camera object.
        
        Returns:
            list: Indices of nodes that are visible.
        """
        visible_indices = []
        for i in candidate_indices:
            if self._is_visible(nodes_positions[i], camera):
                visible_indices.append(i)
        return visible_indices

    def _create_group_gizmo(self, group_center: np.ndarray, selection_count: int):
        """
        Create a group gizmo at the center of selected nodes.
        
        Parameters:
            group_center (np.ndarray): The computed center of the selected nodes.
            selection_count (int): Number of selected nodes (used for scaling).
        
        Returns:
            tuple: The created group gizmo and its initial center.
        """
        group_gizmo = self._server.scene.add_transform_controls(
            name="/group_gizmo",
            opacity=1,
            scale=selection_count,
            disable_rotations=True,
            visible=True,
        )
        group_gizmo.position = tuple(group_center)
        group_initial_center = group_center.copy()
        return group_gizmo, group_initial_center
    
    def _select_nodes(self, indices: list):
        """
        Change the color of selected nodes to green.
        """
        for idx in indices:
            node = self._editor.get_nodes()[idx]
            node.on_selected()
            # node._object.color = (0, 255, 0)
            print(f"Node /sphere_blue_{idx} color set to green (selected).")

    def _update_selection_state(self, visible_indices: list, group_gizmo, group_initial_center: np.ndarray):
        """
        Save the selection state and enable the deselect button.
        
        Parameters:
            visible_indices (list): Indices of the visible (selected) nodes.
            group_gizmo: The group gizmo associated with the selection.
            group_initial_center (np.ndarray): The initial center position of the group.
        """
        self._current_selection["indices"] = visible_indices
        self._current_selection["group_gizmo"] = group_gizmo
        self._current_selection["initial_center"] = group_initial_center
        self._deselect_button.disabled = False

    def _select_click(self, event: viser.GuiEvent) -> None:
        self._select_button.disabled = True
        
        # Register pointer event callback.
        event.client.scene.on_pointer_event(event_type="rect-select")(self._on_rect_selection)
    
    def _deselect_click(self, event: viser.GuiEvent) -> None:
        self._deselect_nodes()
    
    def _deselect_nodes(self) -> None:
        if self._current_selection["group_gizmo"] is not None:
            try:
                self._server.scene.remove_object(self._current_selection["group_gizmo"])
            except Exception:
                self._current_selection["group_gizmo"].visible = False

            if self._current_selection["indices"] is not None:
                for idx in self._current_selection["indices"]:
                    node = self._editor.get_nodes()[idx]
                    node.on_deselected()
                    print(f"Node /sphere_blue_{idx} color reverted to blue (deselected).")

            self._current_selection["group_gizmo"] = None
            self._current_selection["indices"] = None
            self._current_selection["initial_center"] = None
            self._deselect_button.disabled = True
            print("Deselected all points")

    def _group_selection(self, group_gizmo, indices, initial_center):
        """
        On group gizmo move, update the mesh and nodes' positions.
        """
        new_center = np.array(group_gizmo.position)
        delta = new_center - initial_center
        for i in indices:
            node = self._editor.get_nodes()[i]
            new_pos = np.array(node._object.position) + delta
            node._object.position = tuple(new_pos)
            node._control_handle.position = tuple(new_pos)

            distances = np.linalg.norm(self._scene_object.get_global_vertices() - new_pos, axis=1)
            min_index = np.argmin(distances)
            new_local = self._scene_object.get_rotation().inverse().apply(
                new_pos - self._scene_object.get_position()
            )
            self._scene_object.get_mesh().vertices[min_index] = new_local
            self._scene_object.get_global_vertices()[min_index] = new_pos

        # Update the mesh in the scene.
        self._editor.get_scene_object_handle().vertices = self._scene_object.get_mesh().vertices
        
        # Update the initial center for future delta computations.
        initial_center[:] = new_center
        print("Group gizmo moved, updated selected points")

    # def _select_click(self, event: viser.GuiEvent) -> None:
    #     self._select_button.disabled = True
        
    #     # Register pointer event callback.
    #     event.client.scene.on_pointer_event(event_type="rect-select")(self._on_rect_selection)

    # def _deselect_click(self, event: viser.GuiEvent) -> None:
    #     print('Deselected')
    #     self._deselect_button.disabled = True

    #     # self._deselect_nodes()

    # def _on_rect_selection(self, event: viser.ScenePointerEvent) -> None:
    #     # Reset previous selection state.
    #     self._reset_selection_state(event)

    #     camera = event.client.camera

    #     # Get node positions from the editor.
    #     nodes_positions = self._get_nodes_positions()

    #     # Transform the positions to the camera coordinate frame.
    #     sphere_camera = self._transform_to_camera_frame(nodes_positions, camera)

    #     # Project the 3D points to 2D screen space.
    #     proj = self._project_to_screen(sphere_camera, camera)

    #     # Determine the rectangular selection bounds.
    #     rect_min, rect_max = self._get_rectangle_bounds(event.screen_pos)

    #     # Find candidate nodes whose projected positions lie in the rectangle.
    #     candidate_indices = self._get_vertices_in_rect(proj, rect_min, rect_max)

    #     # Filter candidates by checking for occlusion.
    #     visible_indices = self._filter_visible_indices(candidate_indices, nodes_positions, camera)

    #     if not visible_indices:
    #         print("No visible points selected")
    #         return
        
    #     print(f"Selected visible points: {visible_indices}")

    #     # Find the center point to position group gizmo.
    #     selected_positions = nodes_positions[visible_indices]
    #     group_center = selected_positions.mean(axis=0)

    #     # Create a group gizmo at the center of the selected nodes.
    #     group_gizmo, group_initial_center = self._create_group_gizmo(group_center, len(visible_indices))

    #     # Change the color of selected nodes to green.
    #     self._select_nodes(visible_indices)

    #     # Save the selection state and enable the deselect button.
    #     self._update_selection_state(visible_indices, group_gizmo, group_initial_center)

    #     # Register an update callback for the group gizmo.
    #     group_gizmo.on_update(
    #         lambda event: self._group_selection(group_gizmo, visible_indices, group_initial_center)
    #     )

    # def _reset_selection_state(self, event: viser.ScenePointerEvent):
    #     """
    #     Reset the selection state by deselecting nodes, removing the pointer callback,
    #     and enabling the select button.
    #     """
    #     self._deselect_nodes()
    #     event.client.scene.remove_pointer_callback()
    #     self._select_button.disabled = False

    # def _get_nodes_positions(self) -> np.ndarray:
    #     """
    #     Retrieve the current positions of all nodes from the editor.
    #     """
    #     return np.array([np.array(node.get_position()) for node in self._editor.get_nodes()])
    

    # def _transform_to_camera_frame(self, nodes_positions: np.ndarray, camera) -> np.ndarray:
    #     """
    #     Transform node positions from world coordinates to the camera coordinate frame.
    #     """
    #     rows = nodes_positions.shape[0]
    #     ones = np.ones((rows, 1))
    #     node_hom = np.hstack([nodes_positions, ones])
    #     R_camera_world = viser.transforms.SE3.from_rotation_and_translation(
    #         viser.transforms.SO3(camera.wxyz), camera.position
    #     ).inverse()
    #     node_camera = (R_camera_world.as_matrix() @ node_hom.T).T[:, :3]
    #     return node_camera
    
    # def _project_to_screen(self, node_camera: np.ndarray, camera) -> np.ndarray:
    #     """
    #     Project 3D points from the camera frame into 2D screen space.
    #     """
    #     fov, aspect = camera.fov, camera.aspect
    #     proj = node_camera[:, :2] / node_camera[:, 2:3]
    #     proj /= np.tan(fov / 2)
    #     proj[:, 0] /= aspect
    #     proj = (1 + proj) / 2
    #     return proj
    
    # def _get_rectangle_bounds(self, screen_pos: tuple):
    #     """
    #     Compute the min and max corners of the selection rectangle.
    #     """
    #     rect_min = np.minimum(screen_pos[0], screen_pos[1])
    #     rect_max = np.maximum(screen_pos[0], screen_pos[1])
    #     return rect_min, rect_max

    # def _get_vertices_in_rect(
    #     self, proj: np.ndarray, rect_min: np.ndarray, rect_max: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     Determine which projected points lie within the selection rectangle.
    #     """
    #     vertices = np.where(
    #         (proj[:, 0] >= rect_min[0])
    #         & (proj[:, 0] <= rect_max[0])
    #         & (proj[:, 1] >= rect_min[1])
    #         & (proj[:, 1] <= rect_max[1])
    #     )[0]
    #     return vertices
    
    # def _is_visible(self, node_pos: np.ndarray, camera) -> bool:
    #     """
    #     Check if a node at a given world position is visible (i.e. not occluded) from the camera.
    #     """
    #     ray_origin = np.array(camera.position)
    #     ray_dir = node_pos - ray_origin
    #     distance_to_node = np.linalg.norm(ray_dir)
    #     if distance_to_node < 1e-6:
    #         return True

    #     ray_dir_norm = ray_dir / distance_to_node

    #     # Transform ray to local coordinates of the scene object.
    #     ray_origin_local = self._scene_object.get_rotation().inverse().apply(
    #         ray_origin - self._scene_object.get_position()
    #     )
    #     ray_dir_local = self._scene_object.get_rotation().inverse().apply(ray_dir_norm)

    #     intersector = trimesh.ray.ray_triangle.RayMeshIntersector(
    #         self._scene_object.get_mesh()
    #     )
    #     hit_positions, _, _ = intersector.intersects_location(
    #         ray_origin_local.reshape(1, 3),
    #         ray_dir_local.reshape(1, 3),
    #         multiple_hits=False,
    #     )

    #     if len(hit_positions) > 0:
    #         hit_global = (
    #             self._scene_object.get_rotation().apply(hit_positions[0])
    #             + self._scene_object.get_position()
    #         )
    #         hit_distance = np.linalg.norm(hit_global - ray_origin)
    #         # If an intersection is detected before reaching the node, it is occluded.
    #         if hit_distance < distance_to_node - 0.01:
    #             return False
    #     return True

    # def _filter_visible_indices(
    #     self, candidate_indices: np.ndarray, nodes_positions: np.ndarray, camera
    # ) -> list:
    #     """
    #     Filter candidate node indices to include only those that are visible from the camera.
        
    #     Parameters:
    #         candidate_indices (np.ndarray): Array of candidate indices based on the rectangle.
    #         nodes_positions (np.ndarray): Array of all node positions.
    #         camera: Camera object.
        
    #     Returns:
    #         list: Indices of nodes that are visible.
    #     """
    #     visible_indices = []
    #     for i in candidate_indices:
    #         if self._is_visible(nodes_positions[i], camera):
    #             visible_indices.append(i)
    #     return visible_indices
    
    # def _create_group_gizmo(self, group_center: np.ndarray, selection_count: int):
    #     """
    #     Create a group gizmo at the center of selected nodes.
        
    #     Parameters:
    #         group_center (np.ndarray): The computed center of the selected nodes.
    #         selection_count (int): Number of selected nodes (used for scaling).
        
    #     Returns:
    #         tuple: The created group gizmo and its initial center.
    #     """
    #     group_gizmo = self._server.scene.add_transform_controls(
    #         name="/group_gizmo",
    #         opacity=1,
    #         scale=selection_count,
    #         disable_rotations=True,
    #         visible=True,
    #     )
    #     group_gizmo.position = tuple(group_center)
    #     group_initial_center = group_center.copy()
    #     return group_gizmo, group_initial_center
    
    # def _select_nodes(self, indices: list):
    #     """
    #     Change the color of selected nodes to green.
    #     """
    #     for idx in indices:
    #         node = self._editor.get_nodes()[idx]
    #         node.on_selected()
    #         print(f"Node /sphere_blue_{idx} color set to green (selected).")