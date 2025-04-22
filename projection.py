from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CubeProjection(nn.Module):
    """
    Utility class for projecting between equirectangular and cubemap formats.
    """

    def __init__(self):
        super().__init__()
        self.FRONT_FACE     = 0   # +Z
        self.RIGHT_FACE     = 1   # +X
        self.BACK_FACE      = 2   # -Z
        self.LEFT_FACE      = 3   # -X
        self.TOP_FACE       = 4   # +Y
        self.BOTTOM_FACE    = 5   # -Y

    def get_face_coordinates(self, face_idx: int) -> Tuple[float, float, float]:
        """
        Get the direction vector for the center of each face.

        Args:
            face_idx: Face index (0-5)

        Returns:
            Tuple of (x, y, z) coordinates representing the face direction
        """
        if face_idx == self.FRONT_FACE:   return (0, 0, 1)                                      # noqa
        elif face_idx == self.RIGHT_FACE: return (1, 0, 0)                                      # noqa
        elif face_idx == self.BACK_FACE:  return (0, 0, -1)                                     # noqa
        elif face_idx == self.LEFT_FACE:  return (-1, 0, 0)                                     # noqa
        elif face_idx == self.TOP_FACE:   return (0, 1, 0)                                      # noqa
        elif face_idx == self.BOTTOM_FACE: return (0, -1, 0)                                    # noqa
        else: raise ValueError(f"Invalid face index: {face_idx}. Must be between 0 and 5.")     # noqa

    def generate_sampling_grid(
        self,
        height: int,
        width: int,
        fov_degrees: float = 90.0
    ) -> torch.Tensor:
        """
        Generate a sampling grid for a perspective image.

        Args:
            height: Height of the output image
            width: Width of the output image
            fov_degrees: Field of view in degrees

        Returns:
            Tensor of shape (height, width, 2) with normalized coordinates
        """
        fov_rad = math.radians(fov_degrees)

        y = torch.linspace(-1, 1, height)
        x = torch.linspace(-1, 1, width)

        y = y * math.tan(fov_rad / 2)
        x = x * math.tan(fov_rad / 2)

        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

        grid = torch.stack([grid_x, grid_y], dim=-1)

        return grid

    def cube_to_sphere(
        self,
        face_idx: int,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert cube coordinates to spherical coordinates.

        Args:
            face_idx: Face index (0-5)
            grid: Tensor of shape (height, width, 2) with normalized coordinates

        Returns:
            Tensor of shape (height, width, 3) with 3D direction vectors
        """
        x = grid[..., 0]
        y = grid[..., 1]

        if face_idx == self.FRONT_FACE:
            dirs = torch.stack([x, y, torch.ones_like(x)], dim=-1)
        elif face_idx == self.RIGHT_FACE:
            dirs = torch.stack([torch.ones_like(x), y, -x], dim=-1)
        elif face_idx == self.BACK_FACE:
            dirs = torch.stack([-x, y, -torch.ones_like(x)], dim=-1)
        elif face_idx == self.LEFT_FACE:
            dirs = torch.stack([-torch.ones_like(x), y, x], dim=-1)
        elif face_idx == self.TOP_FACE:
            dirs = torch.stack([x, torch.ones_like(x), -y], dim=-1)
        elif face_idx == self.BOTTOM_FACE:
            dirs = torch.stack([x, -torch.ones_like(x), y], dim=-1)
        else:
            raise ValueError(f"Invalid face index: {face_idx}. Must be between 0 and 5.")

        norm = torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs / norm

        return dirs

    def sphere_to_equirect(
        self,
        dirs: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Convert spherical directions to equirectangular coordinates.

        Args:
            dirs: Tensor of shape (..., 3) with 3D direction vectors
            height: Height of the equirectangular image
            width: Width of the equirectangular image

        Returns:
            Tensor of shape (..., 2) with equirectangular coordinates
        """
        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

        # Calculate spherical coordinates
        theta = torch.atan2(x, z)                       # Longitude (azimuth)
        phi = torch.asin(torch.clamp(y, -1.0, 1.0))     # Latitude (elevation)

        # Map to UV coordinates
        u = (theta / (2 * math.pi) + 0.5) * 2 - 1      # Map [-π, π] to [-1, 1]
        v = (phi / (math.pi/2))                         # Map [-π/2, π/2] to [-1, 1]

        grid = torch.stack([u, v], dim=-1)
        return grid

    def equirect_to_sphere(
        self,
        grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert equirectangular coordinates to spherical directions.

        Args:
            grid: Tensor of shape (..., 2) with normalized equirectangular coordinates

        Returns:
            Tensor of shape (..., 3) with 3D direction vectors
        """
        u, v = grid[..., 0], grid[..., 1]

        # Convert from [-1, 1] to spherical coordinates
        theta = u * math.pi                             # [-1, 1] to [-π, π]
        phi = v * (math.pi / 2)                         # [-1, 1] to [-π/2, π/2]

        # Convert to Cartesian coordinates
        x = torch.cos(phi) * torch.sin(theta)
        y = torch.sin(phi)
        z = torch.cos(phi) * torch.cos(theta)

        dirs = torch.stack([x, y, z], dim=-1)
        return dirs

    def equirect_to_cubemap(
        self,
        equirect: torch.Tensor,
        face_size: int,
        overlap_degrees: float = 2.5
    ) -> torch.Tensor:
        """
        Convert an equirectangular panorama to a cubemap representation.

        Args:
            equirect: Tensor of shape (B, C, H, W) with equirectangular images
            face_size: Size of each face (assumed square)
            overlap_degrees: Overlap between adjacent faces in degrees

        Returns:
            Tensor of shape (B, 6, C, face_size, face_size) with cubemap faces
        """
        B, C, H, W = equirect.shape
        device = equirect.device

        fov_degrees = 90.0 + 2 * overlap_degrees

        cubemap = torch.zeros(B, 6, C, face_size, face_size, device=device)

        for face_idx in range(6):
            # Generate sampling grid for this face
            grid = self.generate_sampling_grid(face_size, face_size, fov_degrees).to(device)
            
            # Convert grid coordinates to 3D directions
            dirs = self.cube_to_sphere(face_idx, grid)
            
            # Convert 3D directions to equirectangular coordinates
            eq_grid = self.sphere_to_equirect(dirs, H, W)
            
            # Reshape grid to match the expected format for grid_sample
            # From (H, W, 2) to (B, H, W, 2)
            eq_grid = eq_grid.unsqueeze(0).repeat(B, 1, 1, 1)

            # Sample from equirectangular image
            face = F.grid_sample(
                equirect,
                eq_grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )

            cubemap[:, face_idx] = face

        return cubemap

    def cubemap_to_equirect(
        self,
        cubemap: torch.Tensor,
        height: int,
        width: int
    ) -> torch.Tensor:
        """
        Convert a cubemap representation to an equirectangular panorama.

        Args:
            cubemap: Tensor of shape (B, 6, C, H, W) with cubemap faces
            height: Height of the output equirectangular image
            width: Width of the output equirectangular image

        Returns:
            Tensor of shape (B, C, height, width) with equirectangular panoramas
        """
        B, F, C, face_h, face_w = cubemap.shape
        device = cubemap.device

        equirect = torch.zeros(B, C, height, width, device=device)

        u = torch.linspace(-1, 1, width).to(device)
        v = torch.linspace(-1, 1, height).to(device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing='ij')
        grid = torch.stack([grid_u, grid_v], dim=-1)
        dirs = self.equirect_to_sphere(grid)

        x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]

        eps = 1e-6
        abs_x, abs_y, abs_z = torch.abs(x), torch.abs(y), torch.abs(z)

        face_masks = [
            (z > abs_x - eps) & (z > abs_y - eps),      # (+Z)
            (x > abs_z - eps) & (x > abs_y - eps),      # (+X)
            (-z > abs_x - eps) & (-z > abs_y - eps),    # (-Z)
            (-x > abs_z - eps) & (-x > abs_y - eps),    # (-X)
            (y > abs_x - eps) & (y > abs_z - eps),      # (+Y)
            (-y > abs_x - eps) & (-y > abs_z - eps),    # (-Y)
        ]

        for face_idx in range(6):
            mask = face_masks[face_idx]
            if not mask.any():
                continue

            face_dirs = dirs[mask]

            if face_idx == self.FRONT_FACE:
                face_u = face_dirs[..., 0] / face_dirs[..., 2]
                face_v = face_dirs[..., 1] / face_dirs[..., 2]
            elif face_idx == self.RIGHT_FACE:
                face_u = -face_dirs[..., 2] / face_dirs[..., 0]
                face_v = face_dirs[..., 1] / face_dirs[..., 0]
            elif face_idx == self.BACK_FACE:
                face_u = -face_dirs[..., 0] / face_dirs[..., 2]
                face_v = face_dirs[..., 1] / face_dirs[..., 2]
            elif face_idx == self.LEFT_FACE:
                face_u = face_dirs[..., 2] / face_dirs[..., 0]
                face_v = face_dirs[..., 1] / face_dirs[..., 0]
            elif face_idx == self.TOP_FACE:
                face_u = face_dirs[..., 0] / face_dirs[..., 1]
                face_v = -face_dirs[..., 2] / face_dirs[..., 1]
            elif face_idx == self.BOTTOM_FACE:
                face_u = face_dirs[..., 0] / face_dirs[..., 1]
                face_v = face_dirs[..., 2] / face_dirs[..., 1]

            face_grid = torch.stack([face_u, face_v], dim=-1)

            for b in range(B):
                b_grid = face_grid.unsqueeze(0)

                pixels = F.grid_sample(
                    cubemap[b, face_idx].unsqueeze(0),
                    b_grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                )

                equirect[b, :, mask] = pixels.squeeze(0).squeeze(2)

        return equirect

    def crop_overlapping_regions(
        self,
        faces: torch.Tensor,
        overlap_degrees: float = 2.5
    ) -> torch.Tensor:
        """
        Crop the overlapping regions from the cube faces.

        Args:
            faces: Tensor of shape (B, 6, C, H, W) with cubemap faces
                that include overlapping regions
            overlap_degrees: Overlap between adjacent faces in degrees

        Returns:
            Tensor of shape (B, 6, C, H', W') with cropped faces
        """
        B, F, C, H, W = faces.shape
        device = faces.device

        overlap_ratio = overlap_degrees / 90.0
        overlap_pixels_h = int(H * overlap_ratio)
        overlap_pixels_w = int(W * overlap_ratio)

        new_h = H - 2 * overlap_pixels_h
        new_w = W - 2 * overlap_pixels_w

        cropped_faces = torch.zeros(B, F, C, new_h, new_w, device=device)

        for b in range(B):
            for f in range(F):
                cropped_faces[b, f] = faces[b, f, :,
                                            overlap_pixels_h:H-overlap_pixels_h,
                                            overlap_pixels_w:W-overlap_pixels_w]

        return cropped_faces

    def add_overlapping_regions(
        self,
        faces: torch.Tensor,
        overlap_degrees: float = 2.5
    ) -> torch.Tensor:
        """
        Add overlapping regions to the cube faces.

        Args:
            faces: Tensor of shape (B, 6, C, H, W) with cubemap faces
                without overlapping regions
            overlap_degrees: Overlap between adjacent faces in degrees

        Returns:
            Tensor of shape (B, 6, C, H', W') with faces that include overlapping regions
        """
        B, F, C, H, W = faces.shape

        equirect = self.cubemap_to_equirect(faces, H * 2, W * 4)
        faces_with_overlap = self.equirect_to_cubemap(equirect, H, overlap_degrees)

        return faces_with_overlap

    def forward(
        self,
        x: torch.Tensor,
        input_type: str = 'equirect',
        output_type: str = 'cubemap',
        face_size: Optional[int] = None,
        overlap_degrees: float = 2.5
    ) -> torch.Tensor:
        """
        Convert between different panorama formats.

        Args:
            x: Input tensor (either equirectangular or cubemap)
            input_type: Format of the input tensor ('equirect' or 'cubemap')
            output_type: Format of the output tensor ('equirect' or 'cubemap')
            face_size: Size of the cubemap faces (required if output_type is 'cubemap')
            overlap_degrees: Overlap between adjacent faces in degrees

        Returns:
            Tensor in the specified output format
        """
        if input_type == 'equirect' and output_type == 'cubemap':
            assert face_size is not None, "face_size must be specified when converting to cubemap"
            return self.equirect_to_cubemap(x, face_size, overlap_degrees)
        elif input_type == 'cubemap' and output_type == 'equirect':
            B, _, C, H, W = x.shape
            return self.cubemap_to_equirect(x, H * 2, W * 4)
        elif input_type == output_type:
            return x
        else:
            raise ValueError(f"Invalid conversion: {input_type} to {output_type}. Supported formats are 'equirect' and 'cubemap'.")
