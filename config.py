import os
import json

class CubeDiffConfig:
    def __init__(self, **kwargs):
        # Model architecture
        self.pretrained_model_path = "runwayml/stable-diffusion-v1-5"
        self.latent_channels = 4
        self.model_channels = 320
        self.attention_resolutions = [4, 2, 1]
        self.num_res_blocks = 2
        self.channel_mult = [1, 2, 4, 4]
        self.transformer_depth = 1

        # Cubemap parameters
        self.cube_size = 512
        self.fov = 95.0
        self.overlap = 2.5
        self.effective_fov = 90.0
        self.cube_faces = 6

        # Training parameters
        self.lr = 8e-5
        self.batch_size = 64
        self.max_steps = 30000
        self.warmup_steps = 10000
        self.classifier_free_guidance_prob = 0.1
        self.save_interval = 5000
        self.eval_interval = 1000
        self.gradient_accumulation_steps = 1
        self.mixed_precision = True

        # Diffusion parameters
        self.noise_schedule = "linear"
        self.num_timesteps = 1000
        self.beta_start = 0.00085
        self.beta_end = 0.012
        self.prediction_type = "v"  # v-prediction

        # Inference parameters
        self.inference_steps = 50
        self.guidance_scale = 7.5
        self.seed = 42

        # Paths
        self.output_dir = "outputs"
        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        self.log_dir = os.path.join(self.output_dir, "logs")

        # Dataset
        self.train_datasets = ["polyhaven", "humus", "structured3d", "pano360"]
        self.eval_datasets = ["laval_indoor", "sun360"]

        # Optimizer
        self.optimizer = "adam"
        self.weight_decay = 0.0
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.clip_grad_norm = 1.0

        # Distributed training
        self.local_rank = 0
        self.world_size = 1
        self.distributed = False

        # Update with provided parameters
        self.update(**kwargs)

    def create_dirs(self):
        """Create necessary directories for outputs"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def update(self, **kwargs):
        """Update config parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

    def update_from_file(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __str__(self):
        """String representation of config"""
        config_str = "CubeDiff Configuration:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str
