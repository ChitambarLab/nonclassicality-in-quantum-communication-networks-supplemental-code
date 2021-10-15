import pytest
import numpy as np
from multiple_access_channels import bisender_mac_mutual_info

class TestBisenderMACMutualInfo:
    @pytest.mark.parametrize(
        "mac_behavior, priors_x, priors_y, exp_rates_tuple",
        [
            (
                np.array([[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                np.ones(2) / 2,
                np.ones(2) / 2,
                (0, 0, 0),
            ),
            (
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
                np.ones(2) / 2,
                np.ones(2) / 2,
                (1, 1, 2),
            ),
            (
                np.array(
                    [
                        [1, 0, 0, 1],
                        [0, 1, 1, 0],
                    ]
                ),
                np.ones(2) / 2,
                np.ones(2) / 2,
                (1, 1, 1),
            ),
            (
                np.array(
                    [
                        [1, 1, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 1],
                    ]
                ),
                np.ones(3) / 3,
                np.ones(3) / 3,
                (0.9182958340544896, 0.9182958340544896, 0.9910760598382216),
            ),
            (
                np.array(
                    [
                        [1, 1, 0, 0, 0, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 0, 0, 0, 1],
                    ]
                ),
                np.array([0.5, 0.5, 0]),
                np.array([0, 0.5, 0.5]),
                (1, 1, 1),
            ),
        ],
    )
    def test_bisender_mac_mutual_info(self, mac_behavior, priors_x, priors_y, exp_rates_tuple):
        rates_tuple = bisender_mac_mutual_info(mac_behavior, priors_x, priors_y)
        assert np.allclose(rates_tuple, exp_rates_tuple)
