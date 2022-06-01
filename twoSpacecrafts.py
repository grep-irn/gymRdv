"""
Desc to be added
"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled


class twoSpacecraftsEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self):
        self.GM = 3.986004418e14 # [m3s-2]
        self.earthMeanRadius = 6378136.6
        self.svcMass = 1000 # [kg]

        # Angle at which to fail the episode
        self.altitudeMax = 50000e3
        self.altitudeMin = self.earthMeanRadius + 100e3
        self.maxDistanceFromTarget = 10e3

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        boundaries = np.array(
            [
                self.maxDistanceFromTarget,
                self.maxDistanceFromTarget,
                self.maxDistanceFromTarget,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(-10, 10, (3,), dtype=np.float32)
        self.observation_space = spaces.Box(-boundaries, boundaries, dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.steps_beyond_done = None

    def step(self, action):

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        I_POS_I_SVC, I_VEL_I_SVC, I_POS_I_CLT, I_VEL_I_CLT = self.state
        force = action

        # Servicer dynamics
        altitude = np.linalg.norm(I_POS_I_SVC)
        direction = I_POS_I_SVC / altitude
        xacc = - (direction * self.GM) / (altitude * altitude) + force/self.svcMass

        I_POS_I_SVC = I_POS_I_SVC + self.tau * I_VEL_I_SVC
        I_VEL_I_SVC = I_VEL_I_SVC + self.tau * xacc

        # Client dynamics
        altitude = np.linalg.norm(I_POS_I_CLT)
        direction = I_POS_I_CLT / altitude
        xacc = - (direction * self.GM) / (altitude * altitude)

        I_POS_I_CLT = I_POS_I_CLT + self.tau * I_VEL_I_CLT
        I_VEL_I_CLT = I_VEL_I_CLT + self.tau * xacc

        self.state = (I_POS_I_SVC, I_VEL_I_SVC, I_POS_I_CLT, I_VEL_I_CLT)

        # Relative dynamics
        I_POS_SVC_CLT = I_POS_I_CLT - I_POS_I_SVC
        I_VEL_SVC_CLT = I_VEL_I_CLT - I_VEL_I_SVC

        observations = (I_POS_SVC_CLT, I_VEL_SVC_CLT)

        done = np.array_equal(I_POS_SVC_CLT, 0) & np.array_equal(I_VEL_SVC_CLT, 0)
        reward = 1 if done else 0  # Binary sparse rewards

        return observations, reward, done, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        screen_width = 600
        screen_height = 400

        # world_width = self.x_threshold * 2
        # scale = screen_width / world_width
        # polewidth = 10.0
        # polelen = scale * (2 * self.length)
        # cartwidth = 50.0
        # cartheight = 30.0

        # if self.state is None:
        #     return None

        # I_POS_I_SC = self.state

        # if self.screen is None:
        #     pygame.init()
        #     pygame.display.init()
        #     self.screen = pygame.display.set_mode((screen_width, screen_height))
        # if self.clock is None:
        #     self.clock = pygame.time.Clock()

        # self.surf = pygame.Surface((screen_width, screen_height))
        # self.surf.fill((255, 255, 255))

        # l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        # axleoffset = cartheight / 4.0
        # cartx = I_POS_I_SC[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        # carty = 100  # TOP OF CART
        # cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        # cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        # gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        # gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        # l, r, t, b = (
        #     -polewidth / 2,
        #     polewidth / 2,
        #     polelen - polewidth / 2,
        #     -polewidth / 2,
        # )

        # pole_coords = []
        # for coord in [(l, b), (l, t), (r, t), (r, b)]:
        #     coord = pygame.math.Vector2(coord).rotate_rad(-I_POS_I_SC[2])
        #     coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
        #     pole_coords.append(coord)
        # gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        # gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        # gfxdraw.aacircle(
        #     self.surf,
        #     int(cartx),
        #     int(carty + axleoffset),
        #     int(polewidth / 2),
        #     (129, 132, 203),
        # )
        # gfxdraw.filled_circle(
        #     self.surf,
        #     int(cartx),
        #     int(carty + axleoffset),
        #     int(polewidth / 2),
        #     (129, 132, 203),
        # )

        # gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        # self.surf = pygame.transform.flip(self.surf, False, True)
        # self.screen.blit(self.surf, (0, 0))
        # if mode == "human":
        #     pygame.event.pump()
        #     self.clock.tick(self.metadata["render_fps"])
        #     pygame.display.flip()

        # if mode == "rgb_array":
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        #     )
        # else:
        #     return self.isopen

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False