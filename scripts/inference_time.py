from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
from PIL import Image, ImageDraw
import random
import jax
import jax.numpy as jnp
import numpy as np
import time


DROID_EXAMPLE = {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(6),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }

ER_EXAMPLE =  {
        "state": np.random.rand(12),
        "images": {
            "cam_base": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
            "cam_wrist": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        },
        "prompt": "Pick and place green object in the white basket",
    }

# Run multiple inference calls to observe timing improvements
def time_inference(policy, num_trials=10):
    example = ER_EXAMPLE
    inference_start = time.time()
    for i in range(num_trials):
        start_time = time.time()
        action_chunk = policy.infer(example)["actions"]
        print(action_chunk.shape)
        print(action_chunk[0, :])
        end_time = time.time()
        print(f"Inference {i+1}: {end_time - start_time:.2f} sec")

    inference_end = time.time()
    print(f"Total time taken: {inference_end - inference_start:.2f} s")
    print(f"Average inference time: {1000 * (inference_end - inference_start) / num_trials:.2f} ms")


if __name__ == "__main__":
    # Ensure JAX uses GPU if available
    jax.config.update("jax_platform_name", "gpu")

    # Load configuration and model
    config = config.get_config("pi0_er")
    checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_er")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)

    # Execute timing function
    time_inference(policy)