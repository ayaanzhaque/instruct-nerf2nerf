# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Instruct-NeRF2NeRF configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification

from in2n.data.in2n_datamanager import InstructNeRF2NeRFDataManagerConfig
from in2n.in2n import InstructNeRF2NeRFModelConfig
from in2n.in2n_pipeline import InstructNeRF2NeRFPipelineConfig
from in2n.in2n_trainer import InstructNeRF2NeRFTrainerConfig

in2n_method = MethodSpecification(
    config=InstructNeRF2NeRFTrainerConfig(
        method_name="in2n",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=30000,
        save_only_latest_checkpoint=False,
        mixed_precision=True,
        pipeline=InstructNeRF2NeRFPipelineConfig(
            datamanager=InstructNeRF2NeRFDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=InstructNeRF2NeRFModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF method: https://instruct-nerf2nerf.github.io/",
)
