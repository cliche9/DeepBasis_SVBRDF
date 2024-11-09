import os
import torch
from torch.utils import data as data
from utils import svBRDF, FileClient, imfrombytes, img2tensor, imwrite, tensor2img
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class MatSynthDataPreprocesser:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda')
        brdf_opt = {
            'nbRendering':1,
            'size': 256,
            'split_num': 4,
            'split_axis': 1,
            'concat': True,
            'svbrdf_norm': True,
            'permute_channel': True,
            'order': 'ndrs',
            'lampIntensity': 1
        }        
        self.file_client = FileClient('disk')
        self.renderer = svBRDF(brdf_opt)
        self.init_rendering_settings()

    def resolve_svbrdf(self, svbrdf_folder, save_folder):
        """
        Args:
            str: path to svbrdf folder, for example, /home/zhupengfei/data/datasets/MatSynth_1k/test/

        Returns:
            torch.tensor: stacked svbrdf tensor of shape [n, 3, 256, 1024 (p + ndrs)]
        
        MatSynth_1k dataset folder: .../MatSynth_1k/test/Ceramic/acg_tiles_009/1_basecolor.png
        ├── train
        │ ├── Ceramic
        │ │ ├── acg_tiles_009
        │ │ │ ├── 1_basecolor.png
        │ │ │ ├── 1_diffuse.png
        │ │ │ ├── 1_displacement.png
        │ │ │ ├── 1_height.png
        │ │ │ ├── 1_metallic.png
        │ │ │ ├── 1_normal.png
        │ │ │ ├── 1_roughness.png
        │ │ │ ├── 1_specular.png
        │ │ │ └── ...
        │ │ ├── js_tiles_001
        │ │ └── ...
        │ ├── Concrete
        │ └── ...
        └── test
        """
        
        def resolve_single_map(map_path):
            """Resolve a loaded single texture map.

            Args:
                Str: path to a single texture map.

            Returns:
                Tensor: single texture map tensor of shape [3, 256, 256]
            """
            def numpy_norm(arr, dim=-1):
                length = np.sqrt(np.sum(arr * arr, axis = dim, keepdims=True))
                return arr / (length + 1e-12)


            img_bytes = self.file_client.get(map_path, 'brdf')
            img = imfrombytes(img_bytes, float32=True)[:,:,::-1]

            if suffix == "roughness.png":
                img = np.mean(img, axis=-1, keepdims=True)
            elif suffix == "specular.png" or suffix == "diffuse.png":
                img **= 2.2
            else: # normal.png
                img = numpy_norm(img * 2 - 1, -1) * 0.5 + 0.5

            img = img * 2 - 1

            map = img2tensor(img.copy(),bgr2rgb=False).unsqueeze(0)
            map = F.interpolate(map,size=256,mode='bilinear').squeeze(0)

            return map

        # 定义要读取的文件后缀
        prefixes = [str(i) for i in range(1, 6)]
        suffixes = ["normal.png", "diffuse.png", "roughness.png", "specular.png"]

        # 遍历材料和文件夹
        for material in tqdm(os.listdir(svbrdf_folder)):
            material_path = os.path.join(svbrdf_folder, material)

            assert(os.path.isdir(material_path))

            for texture_folder in os.listdir(material_path):
                texture_path = os.path.join(material_path, texture_folder)

                assert(os.path.isdir(texture_path))

                # 读取并堆叠特定文件后缀的图像
                batch_svbrdfs = []
                for prefix in prefixes:
                    single_maps = []
                    for suffix in suffixes:
                        file_path = os.path.join(texture_path, f"{prefix}_{suffix}")

                        assert(os.path.exists(file_path))

                        map = resolve_single_map(file_path)

                        single_maps.append(map)
                    batch_svbrdfs.append(torch.cat(single_maps, 0))

                batch_svbrdfs = torch.stack(batch_svbrdfs).to(self.device)
                # render one input image from the svbrdf
                rendering_setting = self.get_rendering_setting()
                batch_inputs = self.render_inputs(batch_svbrdfs, *rendering_setting)

                n, d, r, s = torch.split(batch_svbrdfs, [3,3,1,3], dim=1)
                batch_maps = torch.cat([n, d, torch.tile(r, (1,3,1,1)), s], -1)

                for i, prefix in enumerate(prefixes):
                    output_maps = tensor2img(batch_maps[i] * 0.5 + 0.5)
                    output_input = tensor2img(batch_inputs[i] * 0.5 + 0.5)
                    output = np.concatenate((output_input, output_maps), axis=1)
                    output_path = os.path.join(save_folder, material, texture_folder, f"{prefix}.png")
                    imwrite(output, output_path)

    def init_rendering_settings(self):
        # init lighting direction
        surface = self.renderer.surface(384,1.5).to('cpu')
        view_pos = np.array([0,0,self.args.fovZ])
        view_pos= torch.Tensor(view_pos,device="cpu").unsqueeze(0)
        light_dir, view_dir, _, _ = self.renderer.torch_generate(view_pos, view_pos, pos=surface)
        self.light_dir = light_dir.cuda()
        self.view_dir = view_dir.cuda()
        x = 192
        y = 192
        self.sample_light_dir = self.light_dir[:,:,x-128:x+128,y-128:y+128]
        self.sample_view_dir = self.view_dir[:,:,x-128:x+128,y-128:y+128]
        self.sample_light_dis = torch.max(self.sample_view_dir[:,2:3],torch.mean(torch.ones_like(self.sample_view_dir),2,keepdim=True)*0.001)

    def get_rendering_setting(self):
        return self.sample_light_dir, self.sample_view_dir, self.sample_light_dis

    def render_inputs(self, gt_svbrdf, l, v, dis):
        inputs = self.renderer._render(gt_svbrdf, l, v, dis).squeeze(1)
        inputs = torch.clip(inputs,0,1)
        
        return inputs*2-1