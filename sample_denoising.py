import numpy as np
import torch as th
from torch import nn
from PIL import Image
from torch.nn import init
import os, argparse, random
from torchvision import models, transforms
from gen_dif_pri.scripts.imagenet_dataloader.\
    imagenet_dataset import ImageFolderDataset
from gen_dif_pri.scripts.\
    guided_diffusion.script_util_x0 import (
    NUM_CLASSES, args_to_dict, add_dict_to_argparser, 
    create_model_and_diffusion, model_and_diffusion_defaults)

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"set the random seed to {str(seed)}")

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        return ((input - target) ** 2).mean()

class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def forward(self, x):
        b_s, _, h_x, w_x = x.size()
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = th.pow((x[:,:,1:,:] - x[:,:,:h_x-1,:]),2).sum()
        w_tv = th.pow((x[:,:,:,1:] - x[:,:,:,:w_x-1]),2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / b_s

class VGGLoss(nn.Module):
    def __init__(self, layers=9):
        super().__init__()
        self.mse_loss = MSELoss()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1).features[:layers]
        self.model.requires_grad_(False)
        self.model.eval()
    
    def forward(self, input, target):
        batch = th.cat([input, target], dim=0)
        feats = self.model(self.normalize(batch))
        input_feats, target_feats = feats.chunk(2, dim=0)
        return self.mse_loss(input_feats, target_feats)

class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()
        self.disc_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(),
            nn.Conv2d(256, 1, 1), nn.Sigmoid())
        self.optimizer = th.optim.Adam(self.disc_net.parameters(), 1e-3)
    
    def init_disc_net(self):
        for m in self.disc_net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0)
    
    def get_loss_value(self, input, is_real):
        target_value = 1.0 if is_real else 0.0
        target_label = th.ones_like(input) * target_value
        return self.loss(input, target_label)
    
    def forward(self, input, target):
        self.disc_net.train()
        self.optimizer.zero_grad()
        logits = self.disc_net(th.cat((input,target)).detach())
        fake_loss = self.get_loss_value(logits[:input.size(0)], False)
        real_loss = self.get_loss_value(logits[input.size(0):], True)
        (fake_loss + real_loss).backward()
        self.optimizer.step()
        self.disc_net.eval()
        return self.get_loss_value(self.disc_net(input), True)

class GenerativeDegradation(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_dict = None
        self.mse_loss = MSELoss()
        self.perceptual_loss = VGGLoss()
        self.adversarial_loss = GANLoss()
        self.gene_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, 1, 1), nn.LeakyReLU(),
            nn.Conv2d(128, 256, 3, 1, 1), nn.LeakyReLU(),
            nn.Conv2d(256, 3, 1))
        self.optimizer = th.optim.Adam(self.gene_net.parameters(), 1e-3)
    
    def init_gene_net(self):
        for m in self.gene_net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0)
        self.adversarial_loss.init_disc_net()
    
    def forward(self, input, target):
        self.gene_net.train()
        self.optimizer.zero_grad()
        output = self.gene_net(input.detach())
        mse = self.mse_loss(output, input.detach()) * 3.0
        pse = self.perceptual_loss(output, input.detach()) * 2e-3
        adv = self.adversarial_loss(output, target.detach()) * 1e-3
        (mse + pse + adv).backward()
        self.optimizer.step()
        self.gene_net.eval()
        self.loss_dict = {"mse": mse, "pse": pse, "adv": adv}
        return self.gene_net(input)

def main():
    args = create_args()
    device = th.device("cuda")
    os.makedirs(args.result_dir, exist_ok=True)

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    if args.use_fp16: model.convert_to_fp16()
    model.to(device)
    model.eval()

    tv_loss = TVLoss().to(device)
    mse_loss = MSELoss().to(device)
    perceptual_loss = VGGLoss().to(device)
    adversarial_loss = GANLoss().to(device)
    degeneration_model = GenerativeDegradation().to(device)

    def general_cond_fn(x, t, y, x_img=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)

            x_in_hq = ((x_in + 1) / 2).to(th.float32)
            x_tg_lq = ((x_img + 1) / 2).to(th.float32)
            x_in_lq = degeneration_model(x_in_hq, x_tg_lq)

            # First Stage (t ~ [999, 600])
            warmup = 0.5 * (1 - (t[0].item() - 700) / 299)
            mse = mse_loss(x_in_lq, x_tg_lq.detach()) * warmup if t[0] >= 700 else 0

            # Second Stage (t ~ [700, 0])
            mse = mse_loss(x_in_lq[:args.batch_size],
                           x_tg_lq[:args.batch_size].detach()) * 1.0 if t[0] < 700 else mse
            pse = 0 #perceptual_loss(x_in_lq[:args.batch_size],
                                #   x_tg_lq[:args.batch_size].detach()) * 2e-3 if t[0] < 700 else 0
            adv = 0 #adversarial_loss((x_in_hq - x_tg_lq.detach())[:args.batch_size],
                                #    (x_in_hq - x_in_lq).detach()) * 5e-5 if t[0] < 700 else 0
            
            # Third Stage (t ~ [50, 0])
            tv = tv_loss(x_in_hq[:args.batch_size]) * 6e-4 if t[0] < 50 else 0

            loss = args.guidance_scale * (mse + pse + adv + tv)

            guidance_loss_dict = {"mse": mse, "pse": pse, "adv": adv, "tv": tv}

            gui_log, deg_log = "", ""
            for k, v in guidance_loss_dict.items(): gui_log += f"{k}={v:.5f}, "
            for k, v in degeneration_model.loss_dict.items(): deg_log += f"{k}={v:.5f}, "
            print(f"step={t[0]:d}, dege({deg_log[:-2]}), guid(scale={args.guidance_scale:d}, {gui_log[:-2]})")
            
            return th.autograd.grad(-loss, x_in)[0]
    
    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)
    
    dataset = ImageFolderDataset(args.sample_dir, label_file=None)
    dataloader = th.utils.data.DataLoader(dataset, args.batch_size, False)

    num_samples = 0
    for image_lq, label in dataloader:
        init_seed(seed=20)

        image_lq = image_lq.to(device)
        assert image_lq.size(2) == image_lq.size(3)
        assert 256 // image_lq.size(2) in [1, 2, 4]

        adversarial_loss.init_disc_net()
        degeneration_model.init_gene_net()
        image_lq = image_lq.repeat(args.inference_num, 1, 1, 1)
        cond_fn = lambda x, t, y: general_cond_fn(x, t, y, image_lq)

        shape = (image_lq.shape[0], 3, args.image_size, args.image_size)
        model_kwargs = {"y": th.randint(low=0, high=NUM_CLASSES, size=(shape[0],), device=device)}
        sample = diffusion.p_sample_loop(model=model_fn, cond_fn=cond_fn, shape=shape,
                    clip_denoised=args.clip_denoised, model_kwargs=model_kwargs, device=device)
        
        sample = sample[:args.batch_size].detach()
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        for i in range(sample.shape[0]): 
            Image.fromarray(sample[i]).save(os.path.join(args.result_dir, label[i]))
        
        num_samples = num_samples + sample.shape[0]
        print(f"Created {num_samples} samples.")

def create_args():
    result_dir = f"test_samples/Kodak24_256x256/results"
    sample_dir = f"test_samples/Kodak24_256x256/lowlight"
    model_path = f"test_models/256x256_diffusion_uncond.pt"
    defaults = dict(batch_size=1, clip_denoised=True, model_path=model_path)
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--inference_num", type=int, default=2)
    parser.add_argument("--guidance_scale", type=int, default=6000)
    parser.add_argument("--result_dir", type=str, default=result_dir)
    parser.add_argument("--sample_dir", type=str, default=sample_dir)

    return parser.parse_args()

if __name__ == "__main__":
    main()