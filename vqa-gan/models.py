import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImgEncoder(nn.Module):

    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
        #model = models.vgg19(pretrained=True)
        #in_features = model.classifier[-1].in_features  # input size of feature vector
        #model.classifier = nn.Sequential(
        #    *list(model.classifier.children())[:-1])    # remove last fc layer

        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model = nn.Sequential(*(list(model.children())[:-1]))

        self.model = model                              # loaded model without last fc layer
        self.fc = nn.Linear(in_features, embed_size)    # feature vector of image

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        #print('hello')
        with torch.no_grad():
            img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]

        img_feature = img_feature.view(img_feature.size(0), -1)
        intermediate = img_feature
        
        img_feature = self.fc(img_feature)                   # [batch_size, embed_size]

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

        return intermediate, img_feature


class QstEncoder(nn.Module): # for discriminator

    def __init__(self, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        #self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, qst_vec):

        #qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature

class Discriminator(nn.Module):

    def __init__(self, embed_size, ans_vocab_size, word_embed_size, num_layers, hidden_size):

        super(Discriminator, self).__init__()
        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst_emb):

        intermediate, img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst_emb)                 # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return intermediate, combined_feature


class QstAnsEncoder(nn.Module):

    def __init__(self, qst_vocab_size, ans_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstAnsEncoder, self).__init__()
        self.word2vec_qst = nn.Embedding(qst_vocab_size, word_embed_size)
        self.word2vec_ans = nn.Embedding(ans_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question, answer):

        qst_emb = self.word2vec_qst(question)                         # [batch_size, max_qst_length=30, word_embed_size=300]
        ans_vec = self.word2vec_ans(answer)                           # [batch_size, length=1, word_embed_size=300]
        ans_vec = ans_vec.view(ans_vec.size(0), 1, -1)
        qst_vec = torch.cat([ans_vec, qst_emb], dim=1)                # [batch_size, length=31, word_embed_size=300]
        
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=31, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature

    def gen_qst_emb(self, question):
        qst_emb = self.word2vec_qst(question)
        return qst_emb


class ImgDecoder(nn.Module):
    def __init__(self, embed_size, img_feature_size):
        super(ImgDecoder, self).__init__()
        self.image_size = 224
        self.num_channels = 3
        self.noise_dim = 100
        self.embed_size = embed_size
        self.img_feature_size = img_feature_size
        self.latent_dim = self.noise_dim + embed_size + img_feature_size
        self.ngf = 64

		#self.projection = nn.Sequential(
		#	nn.Linear(in_features=self.embed_dim, out_features=self.projected_embed_dim),
		#	nn.BatchNorm1d(num_features=self.projected_embed_dim),
		#	nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#	)

		# based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
        self.netG = nn.Sequential(
			nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 7, 1, 0, bias=False),
			nn.BatchNorm2d(self.ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 7 x 7
			BasicBlock(self.ngf * 8, self.ngf * 8),
            nn.Upsample(scale_factor=2),
			BasicBlock(self.ngf * 8, self.ngf * 4, downsample=conv3x3(self.ngf * 8, self.ngf * 4)),
			##nn.Upsample(scale_factor=2),
			# state size. (ngf*4) x 14 x 14
			BasicBlock(self.ngf * 4, self.ngf * 4),
            nn.Upsample(scale_factor=2),
			BasicBlock(self.ngf * 4, self.ngf * 2, downsample=conv3x3(self.ngf * 4, self.ngf * 2)),
			##nn.Upsample(scale_factor=2),
			# state size. (ngf*2) x 28 x 28
			BasicBlock(self.ngf * 2, self.ngf * 2),
            nn.Upsample(scale_factor=2),
			BasicBlock(self.ngf * 2, self.ngf, downsample=conv3x3(self.ngf * 2, self.ngf)),
			##nn.Upsample(scale_factor=2),
			# state size. (ngf) x 56 x 56
			BasicBlock(self.ngf, self.ngf),
            #BasicBlock(self.ngf, self.ngf),
			nn.Upsample(scale_factor=2),
			# state size. (ngf) x 112 x 112
			BasicBlock(self.ngf, self.ngf),
            nn.ConvTranspose2d(self.ngf, self.ngf, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ngf),
			nn.ReLU(True),
            # state size. (ngf) x 224 x 224
			nn.Conv2d(self.ngf, 3, [3, 3], padding=1),
            #nn.Tanh()
			# state size. (num_channels) x 224 x 224
			)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, embed_vector, img_feature, z):

        # projected_embed = self.projection(embed_vector).unsqueeze(2).unsqueeze(3)
        latent_vector = torch.cat([embed_vector.unsqueeze(2).unsqueeze(3), img_feature.unsqueeze(2).unsqueeze(3), z], 1)
        output = self.netG(latent_vector)
        '''output = (output+1)/0.5

        a=torch.ones(output.size())
        a[:,0]=0.485
        a[:,1]=0.456
        a[:,2]=0.406
        b=torch.ones(output.size())
        b[:,0]=0.299
        b[:,1]=0.224
        b[:,2]=0.225
        a=a.to(device)
        b=b.to(device)

        output = (output-a)/b
        
        #output[:,0] = (((output[:,0]+1)/0.5)-0.485)/0.299
        #output[:,1] = (((output[:,1]+1)/0.5)-0.456)/0.224
        #output[:,2] = (((output[:,2]+1)/0.5)-0.406)/0.225'''

        return output

class Generator(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size, img_feature_size):

        super(Generator, self).__init__()
        self.img_decoder = ImgDecoder(embed_size, img_feature_size)
        self.qstans_encoder = QstAnsEncoder(qst_vocab_size, ans_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)


    def forward(self, question, answer, noise, img_feature):

        qst_feature = self.qstans_encoder(question, answer)        # [batch_size, embed_size]
        output = self.img_decoder(qst_feature, img_feature, noise)

        return output                                              # batch x 3 x 224 x 224

    def gen_qst_emb(self, question):
        qst_emb = self.qstans_encoder.gen_qst_emb(question)
        return qst_emb
