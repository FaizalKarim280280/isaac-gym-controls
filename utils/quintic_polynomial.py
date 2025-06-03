import numpy as np

class QuinticPolynomial:
    def __init__(self, init_pos, init_vel, init_accel, final_pos, final_vel, final_accel, dist, waypoint=None):
        # Derived coefficients
        self.a_0 = init_pos
        self.a_1 = init_vel
        self.a_2 = init_accel / 2.0

        if waypoint is None:
            # Solve the linear equation (Ax = B) without waypoint
            A = np.array([[dist ** 3,      dist ** 4,        dist ** 5],
                          [3 * dist ** 2,   4 * dist ** 3,    5 * dist ** 4],
                          [6 * dist,        12 * dist ** 2,   20 * dist ** 3]])

            B = np.array([final_pos - self.a_0 - self.a_1 * dist - self.a_2 * dist ** 2,
                          final_vel - self.a_1 - 2 * self.a_2 * dist,
                          final_accel - 2 * self.a_2])

            x = np.linalg.solve(A, B)

            self.a_3 = x[0]
            self.a_4 = x[1]
            self.a_5 = x[2]

        else:
            # Incorporate waypoint
            waypoint_pos = waypoint
            waypoint_dist = dist / 2
            A = np.array([
                [waypoint_dist ** 3, waypoint_dist ** 4, waypoint_dist ** 5],
                [dist ** 3, dist ** 4, dist ** 5],
                # [3 * waypoint_dist ** 2, 4 * waypoint_dist ** 3, 5 * waypoint_dist ** 4],
                [3 * dist ** 2, 4 * dist ** 3, 5 * dist ** 4],
                # [6 * waypoint_dist, 12 * waypoint_dist ** 2, 20 * waypoint_dist ** 3],
                [6 * dist, 12 * dist ** 2, 20 * dist ** 3]
            ])

            B = np.array([
                waypoint_pos - self.a_0 - self.a_1 *
                waypoint_dist - self.a_2 * waypoint_dist ** 2,
                final_pos - self.a_0 - self.a_1 * dist - self.a_2 * dist ** 2,
                # 0 - self.a_1 - 2 * self.a_2 * waypoint_dist,  # Zero velocity at waypoint
                final_vel - self.a_1 - 2 * self.a_2 * dist,
                # 0 - 2 * self.a_2,  # Zero acceleration at waypoint
                final_accel - 2 * self.a_2
            ])

            # x = np.linalg.solve(A, B)
            x, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

            self.a_3 = x[0]
            self.a_4 = x[1]
            self.a_5 = x[2]

    def calc_point(self, s):
        xs = self.a_0 + self.a_1 * s + self.a_2 * s ** 2 + \
            self.a_3 * s ** 3 + self.a_4 * s ** 4 + self.a_5 * s ** 5
        return xs

    def calc_first_derivative(self, s):
        xs = self.a_1 + (2 * self.a_2 * s) + (3 * self.a_3 * (s ** 2)) + \
            (4 * self.a_4 * (s ** 3)) + (5 * self.a_5 * (s ** 4))
        return xs

    def calc_second_derivative(self, s):
        xs = 2 * self.a_2 + 6 * self.a_3 * s + 12 * \
            self.a_4 * (s ** 2) + 20 * self.a_5 * (s ** 3)
        return xs

    def calc_third_derivative(self, s):
        xs = 6 * self.a_3 + 24 * self.a_4 * s + 60 * self.a_5 * (s ** 2)
        return xs
