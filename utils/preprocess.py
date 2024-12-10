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
            'size': args.size,
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
        self.size = args.size
        self.init_rendering_settings()

    def resolve_svbrdf(self, svbrdf_folder, save_folder):
        """
        Args:
            str: path to svbrdf folder, for example, /home/zhupengfei/data/datasets/MatSynth_1k/test/

        Returns:
            torch.tensor: stacked svbrdf tensor of shape [n, 3, self.size, self.size * 4 (p + ndrs)]
        
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
                Tensor: single texture map tensor of shape [3, self.size, self.size]
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
            map = F.interpolate(map,size=self.size,mode='bilinear').squeeze(0)

            return map

        # 定义要读取的文件后缀
        prefixes = [str(i) for i in range(1, 6)]
        suffixes = ["normal.png", "diffuse.png", "roughness.png", "specular.png"]

        rendering_setting = self.get_rendering_setting()
        # 遍历材料和文件夹
        for material in tqdm(os.listdir(svbrdf_folder)):
            material_path = os.path.join(svbrdf_folder, material)

            assert(os.path.isdir(material_path))

            texture_folders = os.listdir(material_path)
            tex_batch_size = min(len(texture_folders), 50)
            tex_names = []
            batch_svbrdfs = []

            for i, tex_folder in enumerate(texture_folders):
                texture_path = os.path.join(material_path, tex_folder)

                assert(os.path.isdir(texture_path))

                for prefix in prefixes:
                    single_maps = []
                    for suffix in suffixes:
                        file_path = os.path.join(texture_path, f"{prefix}_{suffix}")

                        assert(os.path.exists(file_path))

                        map = resolve_single_map(file_path)

                        single_maps.append(map)
                    batch_svbrdfs.append(torch.cat(single_maps, 0))
                
                tex_names.append(tex_folder)

                if (i + 1) % tex_batch_size == 0 or i + 1 == len(texture_folders):
                    batch_svbrdfs = torch.stack(batch_svbrdfs).to(self.device)
                    # render one input image from the svbrdf
                    batch_inputs = self.render_inputs(batch_svbrdfs, *rendering_setting)

                    n, d, r, s = torch.split(batch_svbrdfs, [3,3,1,3], dim=1)
                    batch_maps = torch.cat([n, d, torch.tile(r, (1,3,1,1)), s], -1)

                    idx = 0
                    for tex_name in tex_names:
                        for prefix in prefixes:
                            batch_maps[idx] = batch_maps[idx] * 0.5 + 0.5
                            batch_maps[idx][:, :, self.size * 1:self.size * 2] = batch_maps[idx][:, :, self.size * 1:self.size * 2] ** 0.45454
                            batch_maps[idx][:, :, self.size * 3:] = batch_maps[idx][:, :, self.size * 3:] ** 0.45454
                            output_maps = tensor2img(batch_maps[idx])
                            output_input = tensor2img(batch_inputs[idx] * 0.5 + 0.5, gamma=True)
                            output = np.concatenate((output_input, output_maps), axis=1)
                            output_path = os.path.join(save_folder, material, tex_name, f"{prefix}.png")
                            imwrite(output, output_path)
                            idx += 1
                    
                    tex_names = []
                    batch_svbrdfs = []

    def resolve_svbrdf_by_maps(self, svbrdf_folder, save_folder):
        """
        Args:
            str: path to svbrdf_folder, for example, /home/zhupengfei/data/datasets/MatSynth_1k/test/

        Returns:
            torch.tensor: stacked svbrdf tensor of shape [n, 3, self.size, self.size * 4 (p + ndrs)]
        """
        
        def resolve_maps(map_file):
            """Resolve a loaded single texture map.

            Args:
                Str: path to a stacked texture map (ndrs).

            Returns:
                Tensor: single texture map tensor of shape [3, self.size, self.size]
            """
            def numpy_norm(arr, dim=-1):
                length = np.sqrt(np.sum(arr * arr, axis = dim, keepdims=True))
                return arr / (length + 1e-12)

            img_bytes = self.file_client.get(map_file, 'brdf')
            img = imfrombytes(img_bytes, float32=True)[:,:,::-1]

            n, d, r, s = np.array_split(img, 4, axis=1)

            n = numpy_norm(n * 2 - 1, -1) * 0.5 + 0.5
            d **= 2.2
            r = np.mean(r, axis=-1, keepdims=True)
            s **= 2.2

            n = img2tensor((n * 2 - 1).copy(),bgr2rgb=False).unsqueeze(0)
            n = F.interpolate(n,size=self.size,mode='bilinear').squeeze(0)
            d = img2tensor((d * 2 - 1).copy(),bgr2rgb=False).unsqueeze(0)
            d = F.interpolate(d,size=self.size,mode='bilinear').squeeze(0)
            r = img2tensor((r * 2 - 1).copy(),bgr2rgb=False).unsqueeze(0)
            r = F.interpolate(r,size=self.size,mode='bilinear').squeeze(0)
            s = img2tensor((s * 2 - 1).copy(),bgr2rgb=False).unsqueeze(0)
            s = F.interpolate(s,size=self.size,mode='bilinear').squeeze(0)

            return n, d, r, s

        rendering_setting = self.get_rendering_setting()
        # 遍历材料和文件夹

        texture_folders = os.listdir(svbrdf_folder)
        tex_batch_size = min(len(texture_folders), 50)
        
        tex_names = []
        batch_svbrdfs = []
        for i, tex_folder in enumerate(texture_folders):
            tex_name = tex_folder + ".png"

            maps_file = os.path.join(svbrdf_folder, tex_folder, tex_name)

            assert(os.path.exists(maps_file))

            n, d, r, s = resolve_maps(maps_file)

            batch_svbrdfs.append(torch.cat([n, d, r, s], 0))
            
            tex_names.append(tex_name)

            if (i + 1) % tex_batch_size == 0 or i + 1 == len(texture_folders):
                batch_svbrdfs = torch.stack(batch_svbrdfs).to(self.device)
                # render one input image from the svbrdf
                batch_inputs = self.render_inputs(batch_svbrdfs, *rendering_setting)

                n, d, r, s = torch.split(batch_svbrdfs, [3,3,1,3], dim=1)
                batch_maps = torch.cat([n, d, torch.tile(r, (1,3,1,1)), s], -1)

                idx = 0
                for tex_name in tex_names:
                    batch_maps[idx] = batch_maps[idx] * 0.5 + 0.5
                    batch_maps[idx][:, :, self.size * 1:self.size * 2] = batch_maps[idx][:, :, self.size * 1:self.size * 2] ** 0.45454
                    batch_maps[idx][:, :, self.size * 3:] = batch_maps[idx][:, :, self.size * 3:] ** 0.45454
                    output_maps = tensor2img(batch_maps[idx])
                    output_input = tensor2img(batch_inputs[idx] * 0.5 + 0.5, gamma=True)
                    output = np.concatenate((output_input, output_maps), axis=1)
                    output_path = os.path.join(save_folder, tex_name)
                    output_input_path = os.path.join(save_folder, tex_name.replace(".png", "_input.png"))
                    imwrite(output_input, output_input_path)
                    imwrite(output, output_path)
                    idx += 1
                
                tex_names = []
                batch_svbrdfs = []

    def init_rendering_settings(self):
        # init lighting direction
        surface = self.renderer.surface(int(self.size * 1.5),1.5).to('cpu')
        view_pos = np.array([0,0,self.args.viewZ])
        view_pos= torch.Tensor(view_pos,device="cpu").unsqueeze(0)
        light_pos = np.array([0,0,self.args.lightZ])
        light_pos = torch.Tensor(light_pos,device="cpu").unsqueeze(0)
        light_dir, view_dir, _, _ = self.renderer.torch_generate(view_pos, light_pos, pos=surface)
        self.light_dir = light_dir.cuda()
        self.view_dir = view_dir.cuda()
        x = int(self.size * 0.75)
        y = int(self.size * 0.75)
        offset = int(self.size / 2)
        self.sample_light_dir = self.light_dir[:,:,x-offset:x+offset,y-offset:y+offset]
        self.sample_view_dir = self.view_dir[:,:,x-offset:x+offset,y-offset:y+offset]
        self.sample_light_dis = torch.max(self.sample_view_dir[:,2:3],torch.mean(torch.ones_like(self.sample_view_dir),2,keepdim=True)*0.001)

    def get_rendering_setting(self):
        return self.sample_light_dir, self.sample_view_dir, self.sample_light_dis

    def render_inputs(self, gt_svbrdf, l, v, dis):
        inputs = self.renderer._render(gt_svbrdf, l, v, dis).squeeze(1)
        inputs = torch.clip(inputs,0,1)
        
        return inputs*2-1