from data import common as dcommon
from loss.loss_fn import adv
from misc.gpu_utils import parallel_forward as pforward

import torch


class MaskedAdversarial(adv.Adversarial):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loss_d(self, fake, real, mask):
        if real is None:
            d_real = None
        else:
            d_real = pforward(self.discriminator, real, mask)

        d_fake = pforward(self.discriminator, fake, mask)
        loss = self.update_d(d_real, d_fake, mask)
        return loss

    def update_d(self, d_real, d_fake, mask):
        if 'w' in self.name:
            # WGAN loss
            if d_real is None:
                loss = d_fake.mean()
            else:
                loss = d_fake.mean() - d_real.mean()
        else:
            if 'ra' in self.name:
                mean_fake = d_fake.mean()
                mean_real = d_real.mean()
                d_fake -= mean_real
                d_real -= mean_fake

            loss = self.bce(fake=d_fake, real=d_real, mask=mask)

        # Gradient penalty
        if 'gp' in self.name:
            loss = loss + 10 * self.gradient_penalty(d_fake, d_real)

        # Update the discriminator
        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if 'w' in self.name and 'gp' not in self.name:
            # weight clipping for WGAN (not -GP)
            for m in self.modules():
                if isinstance(m, nn.modules.conv._ConvNd):
                    m.weight.data.clamp_(-0.01, 0.01)

        return loss.item()

    def loss_g(self, fake, real, mask):
        d_fake = pforward(self.discriminator, fake, mask)
        if 'w' in self.name:
            loss = -d_fake.mean()
        else:
            if 'ra' in self.name:
                with torch.no_grad():
                    d_real = pforward(self.discriminator, real, mask)
                    mean_real = d_real.mean()

                mean_fake = d_fake.mean()
                d_fake -= mean_real
                d_real -= mean_fake
                loss = self.bce(real=d_fake, fake=d_real, mask=mask)
            else:
                loss = self.bce(real=d_fake, mask=mask)

        return loss

    def forward(self, g, z_d, real_d, fake_g, real_g, mask_d, mask_g):
        if self.training:
            self.loss = 0
            if self.gan_k == 0:
                self.loss = self.loss_d(fake_g.detach(), real_g, mask_g)
            else:
                chunk = lambda x: torch.chunk(x, self.gan_k, dim=0)
                z_chunks = chunk(z_d)
                real_chunks = chunk(real_d)
                mask_chunks = chunk(mask_d)
                for z, real, mask in zip(z_chunks, real_chunks, mask_chunks):
                    with torch.no_grad():
                        fake = pforward(g, z)

                    self.loss += self.loss_d(fake, real, mask)
                # Calculate the average
                self.loss /= self.gan_k
        else:
            self.loss = 0

        # For updating the generator
        if self.training:
            loss_g = self.loss_g(fake_g, real_g, mask_g)
        else:
            loss_g = 0

        return loss_g

    def bce(self, fake=None, real=None, mask=None):
            '''
            A binary cross entropy for masked GANs

            Args:
                fake (Tensor, optional):
                real (Tensor):

            Return:
                Tensor: 
            '''
            if fake is None and real is None:
                raise Exception('You should provide at least one batch')

            scale = mask.size(-1) // real.size(-1)
            mask = dcommon.resize_mask(mask, scale)
            is_class = mask.any(dim=1, keepdim=True)

            if fake is not None:
                s_fake = fake.sigmoid()
                loss_fake = -(1 - s_fake + 1e-12).log()
                loss_fake = loss_fake[is_class].mean()
            else:
                loss_fake = 0

            if real is not None:
                s_real = real.sigmoid()
                loss_real = -(s_real + 1e-12).log()
                loss_real = loss_real[is_class].mean()
            else:
                loss_real = 0

            loss = loss_fake + loss_real
            return loss

