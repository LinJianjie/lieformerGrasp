import torch


class SE3Matrix:
    T_dim = 4
    R_dim = 3
    dof = 6
    eps = 1e-6

    @classmethod
    def exp(cls, epsilon):
        assert epsilon.shape[-1] == cls.dof
        if epsilon.dim() == 1:
            epsilon = epsilon.unsqueeze(0)
        T = torch.eye(cls.T_dim, device=epsilon.device, dtype=epsilon.dtype).unsqueeze(dim=0).repeat(epsilon.shape[0],
                                                                                                     1, 1)
        translation_epsilon = epsilon[:, :3]
        rotation_epsilon = epsilon[:, 3:]
        norm_w = torch.norm(rotation_epsilon, p=2, dim=1)
        norm_w2 = norm_w ** 2
        norm_w3 = norm_w ** 3
        c_norm = torch.cos(norm_w)
        s_norm = torch.sin(norm_w)

        W = SE3Matrix.wedge(rotation_epsilon)
        W2 = torch.bmm(W, W)
        I = torch.eye(cls.R_dim, device=epsilon.device, dtype=epsilon.dtype).unsqueeze(dim=0).repeat(epsilon.shape[0],
                                                                                                     1, 1)
        B = (s_norm / (norm_w + cls.eps)).reshape(W.shape[0], 1, 1).expand_as(W)
        C = ((1 - c_norm) / (norm_w2 + cls.eps)).reshape(W.shape[0], 1, 1).expand_as(W)
        Rotation = I + B * W + C * W2
        D = ((norm_w - s_norm) / (norm_w3 + cls.eps)).reshape(W.shape[0], 1, 1).expand_as(W)
        A = I + C * W + D * W2

        translation = torch.bmm(A, translation_epsilon.unsqueeze(dim=-1))
        T[:, :3, :3] = Rotation
        T[:, :3, 3] = translation.squeeze(dim=-1)
        return T

    @classmethod
    def log(cls, T):
        rotation = T[:, :3, :3]
        translation = T[:, :3, 3]
        cos_angle = (0.5 * cls.trace(rotation) - 0.5).clamp_(-1., 1.)
        angle = cos_angle.acos()
        sin_angle = angle.sin()
        log_R = (angle / (2 * sin_angle + cls.eps)).reshape(rotation.shape[0], 1, 1).expand_as(rotation) * (
                rotation - rotation.transpose(2, 1))
        phi = SE3Matrix.vee(log_R)

        norm_w = torch.norm(phi, p=2, dim=1)
        norm_w2 = norm_w ** 2
        norm_w3 = norm_w ** 3
        c_norm = torch.cos(norm_w)
        s_norm = torch.sin(norm_w)
        W = SE3Matrix.wedge(phi)
        W2 = torch.bmm(W, W)
        I = torch.eye(cls.R_dim, device=phi.device, dtype=phi.dtype).unsqueeze(dim=0).repeat(phi.shape[0],
                                                                                             1, 1)
        C = ((1 - c_norm) / (norm_w2 + cls.eps)).reshape(W.shape[0], 1, 1).expand_as(W)
        D = ((norm_w - s_norm) / (norm_w3 + cls.eps)).reshape(W.shape[0], 1, 1).expand_as(W)
        A = I + C * W + D * W2
        rho = torch.bmm(torch.inverse(A), translation.unsqueeze(dim=-1)).squeeze(dim=-1)
        return torch.cat([rho, phi], dim=-1)

    @classmethod
    def wedge(cls, phi):
        Phi = torch.zeros(phi.shape[0], cls.R_dim, cls.R_dim, device=phi.device, dtype=phi.dtype)
        Phi[:, 0, 1] = -phi[:, 2]
        Phi[:, 1, 0] = phi[:, 2]
        Phi[:, 0, 2] = phi[:, 1]
        Phi[:, 2, 0] = -phi[:, 1]
        Phi[:, 1, 2] = -phi[:, 0]
        Phi[:, 2, 1] = phi[:, 0]
        return Phi

    @classmethod
    def vee(cls, Phi):
        phi = torch.zeros(Phi.shape[0], cls.R_dim, device=Phi.device, dtype=Phi.dtype)
        phi[:, 0] = Phi[:, 2, 1]
        phi[:, 1] = Phi[:, 0, 2]
        phi[:, 2] = Phi[:, 1, 0]
        return phi

    @staticmethod
    def trace(mat):
        tr = (torch.eye(mat.shape[1], dtype=mat.dtype, device=mat.device) * mat).sum(dim=1).sum(dim=1)
        tr.view(mat.shape[0])
        return tr


if __name__ == '__main__':
    pass
