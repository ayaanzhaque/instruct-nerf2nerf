import nerfstudio as ns
import torch
import torch.nn as nn
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path

import nerfstudio as ns
import torch
import torch.nn as nn
from nerfstudio.models.base_model import Model
from nerfstudio.models.nerfacto import NerfactoModel
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path

from PIL import Image
import torchvision.transforms as transforms

import itertools

def find_length_of_original_iterable(cycle_iterator):
    seen = set()
    for item in cycle_iterator:
        if item in seen:
            return len(seen)
        seen.add(item)





def main():
    
    check_point = '/home/kirakiraakira/cs236/instruct-nerf2nerf/outputs/bear/in2n-small/2023-12-04_031034/config.yml'
    
    check_point = Path(check_point)
    config, pipeline, checkpoint_path, _ = eval_setup(check_point)
    model=pipeline.model
    print('pipeline datamanager',pipeline.datamanager)
    print("model outputs for camera ray bundle",model.get_outputs_for_camera_ray_bundle)
    print('pipeline train indices order',pipeline.train_indices_order)
    length_of_original_list = find_length_of_original_iterable(pipeline.train_indices_order)
    print("len of pipeline train indices order:",length_of_original_list)  # This will print 3, the length of my_list
    # print("len of pipeline train indices order:",len(pipeline.train_indices_order))
    # Rest of your code4
    for i in range(length_of_original_list):
        current_spot = next(pipeline.train_indices_order)
        print('spot:',current_spot)
        # get original image from dataset
        original_image = pipeline.datamanager.original_image_batch["image"][current_spot].to(pipeline.device)
        # generate current index in datamanger
        current_index = pipeline.datamanager.image_batch["image_idx"][current_spot]
        print('index:',current_index,current_index.shape)
        
        # get current camera, include camera transforms from original optimizer
        camera_transforms = pipeline.model.camera_optimizer(current_index.unsqueeze(dim=0))
        print('index_to_camera_transforms:',current_index,camera_transforms)
        current_camera = pipeline.datamanager.train_dataparser_outputs.cameras[current_index].to(pipeline.device)
        current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)

        # get current render of nerf
        original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)
        camera_outputs = pipeline.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
        rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2)
        print(i,rendered_image.shape)

        # Save the original image
        original_image_tensor = transforms.ToPILImage()(original_image.squeeze(0).cpu())
        original_image_tensor = original_image_tensor.convert("RGB")
        original_image_path = f"{i}_original_image.png"
        #original_image_tensor.save(original_image_path)

        # Sve the rendered image
        image_tensor = transforms.ToPILImage()(rendered_image.squeeze(0).cpu())  # Remove the batch dimension
        image_tensor = image_tensor.convert("RGB")

        # Save the image
        image_path = str(i)+"_output_image.png"  # You can choose any desired file path and format
        current_index = int(current_index)
        current_index+=1
        formatted_index = f"{current_index:02d}"
        ori_file_path='/home/kirakiraakira/cs236/instruct-nerf2nerf/data/nerfstudio/bear/updated_images/frame_000'+formatted_index +'.jpg'
        image_tensor.save(ori_file_path)
    # current_index = torch.tensor(58)
    # print(current_index.shape)
    # camera_transforms = pipeline.model.camera_optimizer(current_index.unsqueeze(dim=0))
    # print(camera_transforms)


if __name__ == '__main__':
    main()
