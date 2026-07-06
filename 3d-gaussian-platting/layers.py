import torch

def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of quaternions (x, y, z, w) to 3x3 rotation matrices.

    Args:
        quaternions (torch.Tensor): A tensor of shape (N, 4) representing quaternions.
                                     The order is assumed to be (x, y, z, w).

    Returns:
        torch.Tensor: A tensor of shape (N, 3, 3) representing the rotation matrices.
    """
    norm = torch.norm(quaternions, p=2, dim=1, keepdim=True)
    q_norm = quaternions / (norm + 1e-8)

    x, y, z, w = q_norm[:, 0], q_norm[:, 1], q_norm[:, 2], q_norm[:, 3]

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    xw = x * w
    yz = y * z
    yw = y * w
    zw = z * w

    R11 = 1 - 2 * (yy + zz)
    R12 = 2 * (xy - zw)
    R13 = 2 * (xz + yw)

    R21 = 2 * (xy + zw)
    R22 = 1 - 2 * (xx + zz)
    R23 = 2 * (yz - xw)

    R31 = 2 * (xz - yw)
    R32 = 2 * (yz + xw)
    R33 = 1 - 2 * (xx + yy)

    rot_matrices = torch.stack([
        torch.stack([R11, R12, R13], dim=-1),
        torch.stack([R21, R22, R23], dim=-1),
        torch.stack([R31, R32, R33], dim=-1)
    ], dim=-2)

    return rot_matrices

def evaluate_sh_basis(max_sh_degree: int, view_dirs: torch.Tensor) -> torch.Tensor:
    """
    Evaluates spherical harmonic basis functions up to max_sh_degree for a given set of view directions.

    Args:
        max_sh_degree (int): The maximum degree of spherical harmonics to evaluate.
        view_dirs (torch.Tensor): A tensor of shape (N, 3) representing normalized view directions (x, y, z).

    Returns:
        torch.Tensor: A tensor of shape (N, M) where M is the total number of SH coefficients
                      ((max_sh_degree + 1)**2), containing the evaluated SH basis values.
    """
    N = view_dirs.shape[0]
    x, y, z = view_dirs[:, 0], view_dirs[:, 1], view_dirs[:, 2]

    sh_basis_list = []

    # Degree 0 (l=0, m=0)
    sh_basis_list.append(torch.full((N,), 0.28209479177, device=view_dirs.device, dtype=view_dirs.dtype))

    if max_sh_degree > 0:
        # Degree 1 (l=1)
        sh_basis_list.append(-0.4886025119 * y)
        sh_basis_list.append(0.4886025119 * z)
        sh_basis_list.append(-0.4886025119 * x)

    if max_sh_degree > 1:
        # Degree 2 (l=2)
        sh_basis_list.append(1.09254843059 * x * y)
        sh_basis_list.append(-1.09254843059 * y * z)
        sh_basis_list.append(0.31539156525 * (3 * z * z - 1))
        sh_basis_list.append(-1.09254843059 * x * z)
        sh_basis_list.append(0.54627421529 * (x * x - y * y))

    if max_sh_degree > 2:
        # Degree 3 (l=3)
        sh_basis_list.append(-0.5900435899 * y * (3 * x * x - y * y))
        sh_basis_list.append(2.8906114424 * x * y * z)
        sh_basis_list.append(-0.4570457994 * y * (5 * z * z - 1))
        sh_basis_list.append(0.3731760601 * z * (5 * z * z - 3))
        sh_basis_list.append(-0.4570457994 * x * (5 * z * z - 1))
        sh_basis_list.append(1.4453057212 * z * (x * x - y * y))
        sh_basis_list.append(-0.5900435899 * x * (x * x - 3 * y * y))

    sh_basis = torch.stack(sh_basis_list, dim=1)

    return sh_basis
