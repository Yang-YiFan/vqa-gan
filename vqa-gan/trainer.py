import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import os
from tqdm import tqdm

#from txt2image_dataset import Text2ImageDataset
#from models.gan_factory import gan_factory
from misc import Utils, Logger

from data_loader import get_loader
from models import Generator, Discriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['TORCH_HOME'] = './pretrain'

class Trainer(object):
    def __init__(self, args):

        self.input_dir = args.input_dir
        self.log_dir = args.log_dir
        self.model_dir = args.model_dir
        self.max_qst_length = args.max_qst_length
        self.max_num_ans = args.max_num_ans
        self.embed_size = args.embed_size
        self.word_embed_size = args.word_embed_size
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.lr = args.lr
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.save_step = args.save_step
        self.l1_coef = args.l1_coef
        self.l2_coef = args.l2_coef
        self.save_path = args.save_path

        self.noise_dim = 100
        self.beta1 = 0.5
        self.logger = Logger('vqa-gan')
        self.checkpoints_path = 'checkpoints'

        self.data_loader = get_loader(
                                        input_dir = self.input_dir,
                                        input_vqa_train = 'train.npy',
                                        input_vqa_valid = 'valid.npy',
                                        max_qst_length = self.max_qst_length,
                                        max_num_ans = self.max_num_ans,
                                        batch_size = self.batch_size,
                                        num_workers = self.num_workers
                                    )

        qst_vocab_size = self.data_loader['train'].dataset.qst_vocab.vocab_size
        ans_vocab_size = self.data_loader['train'].dataset.ans_vocab.vocab_size


        self.generator = Generator(
                                                            embed_size = self.embed_size,
                                                            qst_vocab_size = qst_vocab_size,
                                                            ans_vocab_size = ans_vocab_size,
                                                            word_embed_size = self.word_embed_size,
                                                            num_layers = self.num_layers,
                                                            hidden_size = self.hidden_size,
                                                            img_feature_size = 512
                                                        ).to(device)
                                                

        self.discriminator = Discriminator(
                                                            embed_size = self.embed_size,
                                                            ans_vocab_size = ans_vocab_size,
                                                            word_embed_size = self.word_embed_size,
                                                            num_layers = self.num_layers,
                                                            hidden_size = self.hidden_size
                                                        ).to(device)


        paramsD = list(self.discriminator.qst_encoder.parameters()) \
                + list(self.discriminator.img_encoder.fc.parameters()) \
                + list(self.discriminator.fc1.parameters()) \
                + list(self.discriminator.fc2.parameters())
        
        self.optimD = torch.optim.Adam(paramsD, lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))


    def train(self):
        criterion = nn.CrossEntropyLoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()
        iteration = 0

        for epoch in range(self.num_epochs):

            running_loss = 0.0
            running_corr_exp1 = 0
            running_corr_exp2 = 0

            for batch_sample in tqdm(self.data_loader['train']):

                iteration += 1

                image = batch_sample['image'].to(device)
                #wrong_image = batch_sample['wrong_image'].to(device)
                question = batch_sample['question'].to(device)
                label = batch_sample['answer_label'].to(device)
                multi_choice = batch_sample['answer_multi_choice']  # not tensor, list.

                '''
                self.logger.draw(image, wrong_image)

                self.optimD.zero_grad()
                self.optimG.zero_grad()

                noise = Variable(torch.randn(image.size(0), 100)).to(device)
                noise = noise.view(noise.size(0), 100, 1, 1)

                output = self.generator(question, label, noise)
                qst_emb = self.generator.gen_qst_emb(question)
                intermediate, prediction = self.discriminator(output, qst_emb)

                loss = criterion(prediction, label)
                '''


                # Train the discriminator
                # add a new loss to discriminator to identify real and fake
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                self.optimG.zero_grad()
                self.optimD.zero_grad()

                qst_emb = self.generator.gen_qst_emb(question)
                activation_real, outputs = self.discriminator(image, qst_emb)
                real_loss = criterion(outputs, label)
                real_score = outputs

                noise = Variable(torch.randn(image.size(0), 100)).to(device)
                noise = noise.view(noise.size(0), 100, 1, 1)

                fake_images = self.generator(question, label, noise, activation_real)
                _, outputs = self.discriminator(fake_images, qst_emb)
                fake_loss = criterion(outputs, label)
                fake_score = outputs

                d_loss = real_loss + fake_loss

                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                self.discriminator.zero_grad()
                self.optimG.zero_grad()
                self.optimD.zero_grad()

                qst_emb = self.generator.gen_qst_emb(question)
                noise = Variable(torch.randn(image.size(0), 100)).to(device)
                noise = noise.view(noise.size(0), 100, 1, 1)

                activation_real ,_ = self.discriminator(image, qst_emb)
                fake_images = self.generator(question, label, noise, activation_real)
                activation_fake, outputs = self.discriminator(fake_images, qst_emb)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                #======= Generator Loss function============
                # This is a customized loss function, the first term is the regular cross entropy loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                #===========================================
                g_loss = criterion(outputs, label) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, image)

                g_loss.backward()
                self.optimG.step()

                if iteration % 5 == 0:
                    self.logger.log_iteration_gan(epoch,d_loss, g_loss, real_score, fake_score)
                    self.logger.draw(image, fake_images)

            self.logger.plot_epoch_w_scores(epoch)

            '''    
                iteration += 1
                right_images = sample['right_images']
                right_embed = sample['right_embed']
                wrong_images = sample['wrong_images']

                right_images = Variable(right_images.float()).to(device)
                right_embed = Variable(right_embed.float()).to(device)
                wrong_images = Variable(wrong_images.float()).to(device)

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).to(device)
                smoothed_real_labels = Variable(smoothed_real_labels).to(device)
                fake_labels = Variable(fake_labels).to(device)

                # Train the discriminator
                self.discriminator.zero_grad()
                outputs, activation_real = self.discriminator(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                noise = Variable(torch.randn(right_images.size(0), 100)).to(device)
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + fake_loss

                d_loss.backward()
                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).to(device)
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_images, right_embed)
                _, activation_real = self.discriminator(right_images, right_embed)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)


                #======= Generator Loss function============
                # This is a customized loss function, the first term is the regular cross entropy loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                #===========================================
                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_images, right_images)

                g_loss.backward()
                self.optimG.step()

                if iteration % 5 == 0:
                    self.logger.log_iteration_gan(epoch,d_loss, g_loss, real_score, fake_score)
                    self.logger.draw(right_images, fake_images)

            self.logger.plot_epoch_w_scores(epoch)

            if (epoch) % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)
            '''

    def predict(self):
        for sample in self.data_loader:
            right_images = sample['right_images']
            right_embed = sample['right_embed']
            txt = sample['txt']

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))

            right_images = Variable(right_images.float()).to(device)
            right_embed = Variable(right_embed.float()).to(device)

            # Train the generator
            noise = Variable(torch.randn(right_images.size(0), 100)).to(device)
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = self.generator(right_embed, noise)

            self.logger.draw(right_images, fake_images)

            for image, t in zip(fake_images, txt):
                im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))
                print(t)







