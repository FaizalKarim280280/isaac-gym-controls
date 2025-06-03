import torch
import roma 

def torch_slerp(init_quat, final_quat, steps):
    q = roma.unitquat_slerp(init_quat, final_quat, steps) # steps, batch, 4
    q = q.permute(1, 0, 2) # batch, steps, 4
    return q

class QuinticPolynomial:
    def __init__(self, 
                 init_pos, 
                 init_vel, 
                 init_accel,
                 final_pos, 
                 final_vel, 
                 final_accel, 
                 dist, 
                 waypoint=None):
        """
        Initialize the quintic polynomial coefficients for batch inputs using PyTorch.

        Parameters:
            init_pos (Tensor): Initial positions of shape (N,).
            init_vel (Tensor): Initial velocities of shape (N,).
            init_accel (Tensor): Initial accelerations of shape (N,).
            final_pos (Tensor): Final positions of shape (N,).
            final_vel (Tensor): Final velocities of shape (N,).
            final_accel (Tensor): Final accelerations of shape (N,).
            dist (float or Tensor): Distance to the goal (scalar or shape (N,)).
            waypoint (Tensor, optional): Waypoint positions of shape (N,). Defaults to None.
        """
        self.a_0 = init_pos
        self.a_1 = init_vel
        self.a_2 = init_accel / 2.0

        if waypoint is None:
            # Solve the linear equation (Ax = B) without waypoint
            A = torch.stack([
                dist ** 3, dist ** 4, dist ** 5,
                3 * dist ** 2, 4 * dist ** 3, 5 * dist ** 4,
                6 * dist, 12 * dist ** 2, 20 * dist ** 3
            ], dim=-1).reshape(-1, 3, 3)  # (N, 3, 3)
            B = torch.stack([
                final_pos - self.a_0 - self.a_1 * dist - self.a_2 * dist ** 2,
                final_vel - self.a_1 - 2 * self.a_2 * dist,
                final_accel - 2 * self.a_2
            ], dim=-1)  # (N, 3)

            X = torch.linalg.solve(A, B)  # (N, 3)
            self.a_3 = X[:, 0]
            self.a_4 = X[:, 1]
            self.a_5 = X[:, 2]

        else:
            # Incorporate waypoint
            waypoint_dist = dist / 2

            A = torch.stack([
                waypoint_dist ** 3, waypoint_dist ** 4, waypoint_dist ** 5,
                dist ** 3, dist ** 4, dist ** 5,
                3 * dist ** 2, 4 * dist ** 3, 5 * dist ** 4,
                6 * dist, 12 * dist ** 2, 20 * dist ** 3
            ], dim=-1).reshape(-1, 3, 3)  # (N, 3, 3)

            B = torch.stack([
                waypoint - self.a_0 - self.a_1 * waypoint_dist - self.a_2 * waypoint_dist ** 2,
                final_pos - self.a_0 - self.a_1 * dist - self.a_2 * dist ** 2,
                final_vel - self.a_1 - 2 * self.a_2 * dist
            ], dim=-1)  # (N, 3)

            X, _, _, _ = torch.linalg.lstsq(A, B)
            self.a_3 = X[:, 0]
            self.a_4 = X[:, 1]
            self.a_5 = X[:, 2]

    def calc_point(self, s):
        """
        Calculate position at scalar or tensor of distances `s`.

        Parameters:
            s (Tensor): Time points of shape (T,) or (N, T).

        Returns:
            Tensor: Positions of shape (N, T).
        """
        return (self.a_0.unsqueeze(-1) +
                self.a_1.unsqueeze(-1) * s +
                self.a_2.unsqueeze(-1) * s ** 2 +
                self.a_3.unsqueeze(-1) * s ** 3 +
                self.a_4.unsqueeze(-1) * s ** 4 +
                self.a_5.unsqueeze(-1) * s ** 5)

    def calc_first_derivative(self, s):
        """
        Calculate velocity at scalar or tensor of distances `s`.

        Parameters:
            s (Tensor): Time points of shape (T,) or (N, T).

        Returns:
            Tensor: Velocities of shape (N, T).
        """
        return (self.a_1.unsqueeze(-1) +
                2 * self.a_2.unsqueeze(-1) * s +
                3 * self.a_3.unsqueeze(-1) * s ** 2 +
                4 * self.a_4.unsqueeze(-1) * s ** 3 +
                5 * self.a_5.unsqueeze(-1) * s ** 4)

    def calc_second_derivative(self, s):
        """
        Calculate acceleration at scalar or tensor of distances `s`.

        Parameters:
            s (Tensor): Time points of shape (T,) or (N, T).

        Returns:
            Tensor: Accelerations of shape (N, T).
        """
        return (2 * self.a_2.unsqueeze(-1) +
                6 * self.a_3.unsqueeze(-1) * s +
                12 * self.a_4.unsqueeze(-1) * s ** 2 +
                20 * self.a_5.unsqueeze(-1) * s ** 3)

    def calc_third_derivative(self, s):
        """
        Calculate jerk at scalar or tensor of distances `s`.

        Parameters:
            s (Tensor): Time points of shape (T,) or (N, T).

        Returns:
            Tensor: Jerks of shape (N, T).
        """
        return (6 * self.a_3.unsqueeze(-1) +
                24 * self.a_4.unsqueeze(-1) * s +
                60 * self.a_5.unsqueeze(-1) * s ** 2)