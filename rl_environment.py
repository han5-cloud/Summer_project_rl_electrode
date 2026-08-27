"""Reproducible synthetic RGB--relative-height inspection environment.

This version keeps the four-action and seven-observation task used by the
existing PPO model, but makes the terminology and randomisation explicit:

* the second image channel is a directly generated *relative-height* array,
  not raw stereo-camera depth;
* height values are dimensionless synthetic relative-height units (SRHU), not
  millimetres;
* an explicit seed reproduces both the procedural scene and starting window;
* subsequent resets use a deterministic sequence of different scene seeds.

The procedural ground-truth mask is hidden from the policy observation.  It is
used only to calculate reward and evaluation outcomes inside the simulator.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import uniform_filter


WINDOW_SIZE = 100
BIG_SIZE = 600
CONVEYOR_STEP = 35
CAMERA_STEP = 35
STOP_ACTION = 3
CONFIRM_THRESHOLD = 0.12

# Dimensionless synthetic relative-height parameters.  These are engineering
# assumptions for the simulator and are not measurements of coating thickness.
HEIGHT_BACKGROUND_NOISE_STD = 0.01
RESIDUE_HEIGHT_MIN = 0.10
RESIDUE_HEIGHT_MAX = 0.50
HEIGHT_DETECTION_THRESHOLD = 0.10
HEIGHT_UNIT_LABEL = "synthetic relative-height unit (SRHU)"


# Precompute deterministic RGB background noise to reduce reset cost.
_NOISE_POOL_SIZE = 25
_background_rng = np.random.RandomState(12345)
_noise_pool: list[np.ndarray] = []
for _ in range(_NOISE_POOL_SIZE):
    background = np.tile(
        np.array([50, 52, 55], dtype=np.float32),
        (BIG_SIZE, BIG_SIZE, 1),
    )
    background += _background_rng.normal(
        0, 4, size=(BIG_SIZE, BIG_SIZE, 3)
    ).astype(np.float32)
    _noise_pool.append(np.clip(background, 0, 255).astype(np.uint8))


def make_synthetic_electrode(
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate RGB, relative-height and procedural ground-truth arrays.

    Three to six elliptical residue regions are generated.  The same latent
    ellipse geometry defines the ground-truth region and where RGB variance and
    positive relative height are added.  Overlapping ellipses use the maximum
    latent height instead of adding heights repeatedly, so overlap does not
    represent an undefined stack of multiple coating layers.
    """

    rng = np.random.RandomState(seed)
    height, width = BIG_SIZE, BIG_SIZE

    base = _noise_pool[rng.randint(0, _NOISE_POOL_SIZE)]
    rgb_image = base.astype(np.float32).copy()

    ground_truth_mask = np.zeros((height, width), dtype=bool)
    latent_residue_height = np.zeros((height, width), dtype=np.float32)

    number_of_patches = rng.randint(3, 7)
    for _ in range(number_of_patches):
        centre_x = rng.randint(100, width - 100)
        centre_y = rng.randint(100, height - 100)
        radius_x = rng.randint(40, 80)
        radius_y = rng.randint(30, 60)

        x0, x1 = max(0, centre_x - radius_x), min(width, centre_x + radius_x)
        y0, y1 = max(0, centre_y - radius_y), min(height, centre_y + radius_y)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        local_mask = (
            (xx - centre_x) ** 2 / radius_x**2
            + (yy - centre_y) ** 2 / radius_y**2
        ) < 1

        rgb_image[y0:y1, x0:x1][local_mask] = np.array(
            [52, 50, 48]
        ) + rng.normal(0, 12, size=(local_mask.sum(), 3))
        ground_truth_mask[y0:y1, x0:x1] |= local_mask

        patch_height = rng.uniform(
            RESIDUE_HEIGHT_MIN,
            RESIDUE_HEIGHT_MAX,
            size=local_mask.sum(),
        ).astype(np.float32)
        height_region = latent_residue_height[y0:y1, x0:x1]
        height_region[local_mask] = np.maximum(
            height_region[local_mask], patch_height
        )

    background_height_noise = rng.normal(
        0,
        HEIGHT_BACKGROUND_NOISE_STD,
        size=(height, width),
    ).astype(np.float32)
    relative_height_map = latent_residue_height + background_height_noise

    return (
        np.clip(rgb_image, 0, 255).astype(np.uint8),
        relative_height_map,
        ground_truth_mask,
    )


def local_std(gray_image: np.ndarray, size: int = 7) -> np.ndarray:
    """Return local image-intensity standard deviation (texture proxy)."""

    image = gray_image.astype(np.float32)
    local_mean = uniform_filter(image, size=size)
    local_mean_square = uniform_filter(image**2, size=size)
    local_variance = np.maximum(local_mean_square - local_mean**2, 0)
    return np.sqrt(local_variance)


ACTIONS = {0: -CAMERA_STEP, 1: CAMERA_STEP, 2: 0}


class ReproducibleElectrodeInspectionEnv(gym.Env):
    """Four-action inspection environment with a seven-feature observation."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 15, base_seed: int = 0):
        super().__init__()
        self.max_steps = int(max_steps)
        self.base_seed = int(base_seed)
        self._scene_seed_generator = np.random.default_rng(self.base_seed)
        self._has_reset = False

        self.rgb_image: np.ndarray | None = None
        self.relative_height_map: np.ndarray | None = None
        self.ground_truth_mask: np.ndarray | None = None
        self.current_scene_seed: int | None = None

        self.x = 0
        self.y = 0
        self.steps_taken = 0

        self.action_space = spaces.Discrete(4)
        # [x, y, RGB texture ratio, mean texture, mean relative height,
        #  relative-height threshold ratio, elapsed-step ratio]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None):
        """Reset reproducibly while generating a different scene each episode.

        Passing ``seed`` reproduces that exact scene and starting position.
        With no explicit seed, a deterministic sequence derived from
        ``base_seed`` is used.
        """

        if seed is not None:
            effective_seed = int(seed)
            super().reset(seed=effective_seed)
            self._scene_seed_generator = np.random.default_rng(effective_seed)
            scene_seed = effective_seed
        elif not self._has_reset:
            super().reset(seed=self.base_seed)
            scene_seed = self.base_seed
        else:
            super().reset(seed=None)
            scene_seed = int(
                self._scene_seed_generator.integers(0, np.iinfo(np.int32).max)
            )

        self.current_scene_seed = scene_seed
        (
            self.rgb_image,
            self.relative_height_map,
            self.ground_truth_mask,
        ) = make_synthetic_electrode(scene_seed)

        self.x = int(self.np_random.integers(0, BIG_SIZE - WINDOW_SIZE + 1))
        self.y = 0
        self.steps_taken = 0
        self._has_reset = True

        observation = self._get_observation()
        info = {"scene_seed": self.current_scene_seed}
        return observation, info

    def _current_rgb_view(self) -> np.ndarray:
        assert self.rgb_image is not None
        return self.rgb_image[
            self.y : self.y + WINDOW_SIZE,
            self.x : self.x + WINDOW_SIZE,
        ]

    def _current_height_view(self) -> np.ndarray:
        assert self.relative_height_map is not None
        return self.relative_height_map[
            self.y : self.y + WINDOW_SIZE,
            self.x : self.x + WINDOW_SIZE,
        ]

    def _rgb_texture_features(self) -> tuple[float, float]:
        view = self._current_rgb_view()
        grayscale = np.dot(view[..., :3], [0.2989, 0.5870, 0.1140])
        texture_map = local_std(grayscale, size=7)
        threshold = texture_map.mean() + 1.2 * texture_map.std()
        texture_mask = texture_map > threshold
        return float(texture_mask.mean()), float(texture_map.mean())

    def _relative_height_features(self) -> tuple[float, float]:
        height_view = self._current_height_view()
        height_mask = height_view > HEIGHT_DETECTION_THRESHOLD
        return float(height_mask.mean()), float(height_view.mean())

    def _true_residue_ratio(self) -> float:
        assert self.ground_truth_mask is not None
        mask_view = self.ground_truth_mask[
            self.y : self.y + WINDOW_SIZE,
            self.x : self.x + WINDOW_SIZE,
        ]
        return float(mask_view.mean())

    def _get_observation(self) -> np.ndarray:
        texture_ratio, mean_texture = self._rgb_texture_features()
        height_ratio, mean_height = self._relative_height_features()

        observation = np.array(
            [
                self.x / (BIG_SIZE - WINDOW_SIZE),
                self.y / (BIG_SIZE - WINDOW_SIZE),
                texture_ratio,
                np.clip(mean_texture / 20.0, 0, 1),
                np.clip(mean_height / RESIDUE_HEIGHT_MAX, 0, 1),
                height_ratio,
                self.steps_taken / self.max_steps,
            ],
            dtype=np.float32,
        )
        return observation

    def step(self, action: int):
        action = int(action)
        self.steps_taken += 1
        terminated = False
        truncated = False

        if action == STOP_ACTION:
            true_ratio = self._true_residue_ratio()
            reward = 5.0 - 0.1 * self.steps_taken if true_ratio > CONFIRM_THRESHOLD else -3.0
            terminated = True
        else:
            self.x = int(
                np.clip(
                    self.x + ACTIONS.get(action, 0),
                    0,
                    BIG_SIZE - WINDOW_SIZE,
                )
            )
            self.y += CONVEYOR_STEP
            reward = -0.05

            if self.y >= BIG_SIZE - WINDOW_SIZE:
                self.y = BIG_SIZE - WINDOW_SIZE
                reward -= 1.0
                truncated = True

            if self.steps_taken >= self.max_steps and not truncated:
                reward -= 2.0
                truncated = True

        observation = self._get_observation()
        info = {
            "scene_seed": self.current_scene_seed,
            "true_residue_ratio": self._true_residue_ratio(),
        }
        return observation, float(reward), terminated, truncated, info


if __name__ == "__main__":
    environment = ReproducibleElectrodeInspectionEnv(max_steps=15, base_seed=101)
    first_observation, first_info = environment.reset(seed=1234)
    second_observation, second_info = environment.reset(seed=1234)
    print("Observation shape:", first_observation.shape)
    print("Same explicit seed reproduces observation:", np.array_equal(first_observation, second_observation))
    print("Scene seeds:", first_info["scene_seed"], second_info["scene_seed"])
