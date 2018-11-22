import numpy as np
import cPickle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy.random import shuffle
import sys
import os
import csv
import argparse
import time

np.random.seed(1234)

# nb_words = 500000000
# MAX_SEQUENCE_LENGTH=77
# MAX_ASPECTS=13
# MAX_LEN_ASPECT=5
# EMBEDDING_DIM = 300
# HIDDEN_DIM = 300
# OUTPUT_DIM = 350
# HOP_SIZE = 15
# BATCH_SIZE = 50
# NB_EPOCH = 50

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='does not use GPU')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, metavar='L2',
                    help='L2 regularization weight')
parser.add_argument('--batch-size', type=int, default=25, metavar='BS',
                    help='batch size')
parser.add_argument('--epochs', type=int, default=30, metavar='E',
                    help='number of epochs')
parser.add_argument('--hops', type=int, default=10, metavar='H',
                    help='number of hops')
parser.add_argument('--hidden-size', type=int, default=400, metavar='HS',
                    help='hidden size')
parser.add_argument('--output-size', type=int, default=400, metavar='OS',
                    help='output size')
parser.add_argument('--dropout-p', type=float, default=0.5, metavar='DO1',
                    help='embedding dropout')
parser.add_argument('--dropout-lstm', type=float, default=0.1, metavar='DO2',
                    help='lstm dropout')
parser.add_argument('--dataset', default='Restaurants', metavar='D',
                    help='Laptop or Restaurants')
args = parser.parse_args()
print args
HIDDEN_DIM          = args.hidden_size
OUTPUT_DIM          = args.output_size
HOP_SIZE            = args.hops
BATCH_SIZE          = args.batch_size
NB_EPOCH            = args.epochs
nb_words            = 500000000
MAX_SEQUENCE_LENGTH = 77 if args.dataset=='Laptop' else 69
MAX_ASPECTS         = 13
MAX_LEN_ASPECT      = 5 if args.dataset=='Laptop' else 19
EMBEDDING_DIM       = 300

class PreProcessing():

    def __init__(self, tr_data, te_data, tokenizer, batch_size):
        self.tag_to_ix = {"positive": 0, "negative": 1, "neutral": 2}
        self.tokenizer = tokenizer # Tokenizer(num_words=nb_words)
        self.sents=zip(*tr_data)[0]
        self.sents1=zip(*te_data)[0]
        self.labels=zip(*tr_data)[3]
        self.aspects=zip(*tr_data)[1]
        self.aspect=zip(*tr_data)[2]
        self.batch_size=batch_size

    def prepare_sequence(self, seq, to_ix):
        return [to_ix[w] for w in seq]

    def keras_data_prepare(self, fit=True):
        if fit:
            self.tokenizer.fit_on_texts(self.sents+self.sents1)
        sequences = self.tokenizer.texts_to_sequences(self.sents)
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        return data

    def return_vars(self):
        return self.tokenizer

    def prepare_data(self, data, batch_id, word_embeddings):
        aspect_sequence=[]
        limit = [batch_id*self.batch_size, (batch_id+1)*self.batch_size]
        for item in self.aspects[limit[0]:limit[1]]:
            temp=self.tokenizer.texts_to_sequences(item)
            aspect_sequence.append(temp)
        aspect_ = self.tokenizer.texts_to_sequences(list(self.aspect[limit[0]:limit[1]]))
        train_temp=[]
        j=0
        for datam in data[limit[0]:limit[1]]:
            train_temp.append([datam,aspect_sequence[j],aspect_[j],self.labels[limit[0]:limit[1]][j]])
            j=j+1
        training_data_x0=[]
        training_data_x1=[]
        training_data_y=[]
        attention_mat2 =[]
        attention_mat = []
        for item1 in train_temp:
            sent, aspects, aspect, sentiment = item1[0], item1[1], item1[2], item1[3]
            att = []
            for i in range(0,len(sent)):
                if sent[i] == 0:
                    att.append(0)
                else:
                    att.append(1)

            att_tensor = autograd.Variable(torch.FloatTensor(att) if not args.cuda else torch.cuda.FloatTensor(att),requires_grad=False)

            temp_mask_sent = att_tensor.view(att_tensor.size()[0],-1).expand(-1, 2*EMBEDDING_DIM)
            att_tensor = att_tensor.unsqueeze(0)
            tensor = torch.LongTensor(sent) if not args.cuda else torch.cuda.LongTensor(sent)
            sent1=autograd.Variable(tensor)

            aspects1=[]
            for item in aspects:
                temp = torch.LongTensor(item) if not args.cuda else torch.cuda.LongTensor(item)
                temp = autograd.Variable(temp)
                temp = word_embeddings(temp)
                temp = torch.mean(temp,dim=0)
                aspects1.append(temp)

            aspect = torch.LongTensor(aspect) if not args.cuda else torch.cuda.LongTensor(aspect)
            aspect = autograd.Variable(aspect)

            label=self.prepare_sequence(sentiment, self.tag_to_ix)

            embeds=word_embeddings(sent1)

            #aspect = torch.LongTensor(aspect)
            #aspect = autograd.Variable(aspect)
            aspect1= word_embeddings(aspect)
            aspect1= torch.mean(aspect1,dim=0)
            aspect1 = aspect1.expand(len(sent),-1)

            sepr = []
            att2 = []
            for i in range(0,MAX_ASPECTS-len(aspects)):
                sepr.append(autograd.Variable(torch.zeros((MAX_SEQUENCE_LENGTH,2*EMBEDDING_DIM)).type(ftype).unsqueeze(0)))
                att2.append(0)

            for item in aspects1:
                item = item.expand(len(sent),-1)
                sepr.append(torch.mul(torch.cat([embeds,item],dim=1),temp_mask_sent).unsqueeze(0))
                att2.append(1)

            aspect1 = torch.mul(torch.cat([embeds,aspect1],dim=1),temp_mask_sent)

            att2_tensor = autograd.Variable(torch.FloatTensor(att2) if not args.cuda else torch.cuda.FloatTensor(att2),requires_grad=False).unsqueeze(0)
            sepr_tensor=torch.cat(sepr,dim=0)
            sepr_tensor = sepr_tensor.unsqueeze(0)
            training_data_x0.append(sepr_tensor)
            training_data_x1.append(aspect1.unsqueeze(0))
            training_data_y.append(label)
            attention_mat2.append(att2_tensor)
            attention_mat.append(att_tensor)

        att2_var = torch.cat(attention_mat2,dim=0)
        att_var = torch.cat(attention_mat, dim =0 )
        return torch.cat(training_data_x0,dim=0), torch.cat(training_data_x1,dim=0), autograd.Variable(torch.LongTensor(to_categorical(training_data_y,3)) if not args.cuda else torch.cuda.LongTensor(to_categorical(training_data_y,3))),att2_var, att_var

class AttnRNN(nn.Module):
    def __init__(self, hop_size, batch_size, input_size, sent_size, output_size,
            dropout_p=args.dropout_p, dropout_lstm = args.dropout_lstm,
            max_length=MAX_SEQUENCE_LENGTH):
        super(AttnRNN, self).__init__()
        self.hop_size = hop_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.sent_size = sent_size
        self.dropout_p = dropout_p
        self.dropout_lstm = dropout_lstm
        self.max_length = max_length
        self.hidden_sentence_gru = self.init_hidden2(self.batch_size)
        self.hidden_aspect_gru = self.init_hidden(self.batch_size)
        self.hidden_aspect_write_gru=self.init_hidden(self.batch_size)
        #self.hidden_aspect_repr_gru = self.init_aspect_hidden(self.batch_size)
        self.sentence_gru = nn.GRU(self.input_size*2, self.sent_size)
        self.aspect_gru = nn.GRU(self.sent_size, self.output_size)
        self.aspect_write_gru = nn.GRU(self.output_size, self.output_size)
        # self.aspect_write_gru = nn.GRU(self.output_size, self.output_size/2,
        #         bidirectional=True)
        #self.aspect_repr_gru = nn.GRU(self.input_size*2, self.sent_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_lstm)
        self.attn = nn.Linear(self.sent_size, 1)
        self.attn2 = nn.Linear(1, 1)
        self.affine = nn.Linear(self.output_size,3)
        self.dimproj = nn.Linear(self.sent_size, self.output_size)

    def forward(self, sents, aspects, attention_mat1, attention_mat2, batch_size):
        sents=sents.permute(1,2,0,3) # -> (aspect, seq, batch, embed*2)
        outputs = []
        alphas=[]
        for sent_asp in sents:
            embedded = self.dropout(sent_asp)
            output, hidden_sentence_gru = self.sentence_gru(embedded, self.hidden_sentence_gru)
            #print attention_mat1.size()
            temp_attention_mat1 = attention_mat1.view(attention_mat1.size()[0],attention_mat1.size()[1],1).expand(-1,-1,output.size()[2])
            #print temp_attention_mat1.size()
            #sys.exit(1)
            output = torch.mul(output.permute(1,0,2),temp_attention_mat1)
            output = self.dropout2(output)
            #print output.size()
           # sys.exit(1)
            attn_weights = F.softmax(
                self.attn(output.permute(1,0,2)), dim=0)
            #print attn_weights.size()
            #print attention_mat1.size()
            #sys.exit(1)
            masked_attn_weights = torch.mul(attn_weights.squeeze().permute(1,0),attention_mat1)
            #print masked_attn_weights.size()
            _sums = masked_attn_weights.sum(-1).unsqueeze(1).expand(-1,masked_attn_weights.size()[1])
            #print _sums.size()
            attentions = masked_attn_weights.div(_sums).unsqueeze(1).permute(2,0,1)
            alphas.append(attentions.permute(1,2,0).unsqueeze(0))

            #print attentions.permute(1,0,2).squeeze()[47].sum()
            #print attn_weights.permute(1,0,2)
            attn_applied = torch.bmm(attentions.permute(1,2,0),
                                 output).squeeze()
            output = F.relu(attn_applied)
            outputs.append(output.unsqueeze(0))

        aspec_rep = torch.cat(outputs, dim=0)
        output, hidden_aspect_gru = self.aspect_gru(aspec_rep,self.hidden_aspect_gru)

        temp_attention_mat2 = attention_mat2.view(attention_mat2.size()[0],attention_mat2.size()[1],1).expand(-1,-1,output.size()[2])

        output = torch.mul(output.permute(1,0,2),temp_attention_mat2)
        output = self.dropout2(output)

        aspects = aspects.permute(1,0,2)
        outputa_,hida_ = self.sentence_gru(aspects,self.hidden_sentence_gru)
        temp_attention_mat3 = attention_mat1.view(attention_mat1.size()[0],attention_mat1.size()[1],1).expand(-1,-1,outputa_.size()[2])
        outputa_ = torch.mul(outputa_.permute(1,0,2),temp_attention_mat3)
        attn_weights_ = F.softmax(
                self.attn(outputa_.permute(1,0,2)), dim=0)
        masked_attn_weights_ = torch.mul(attn_weights_.squeeze().permute(1,0),attention_mat1)
        _sums_ = masked_attn_weights_.sum(-1).unsqueeze(1).expand(-1,masked_attn_weights_.size()[1])
        attentions_ = masked_attn_weights_.div(_sums_).unsqueeze(1).permute(2,0,1)
        attn_applied_ = torch.bmm(attentions_.permute(1,2,0),
                                 outputa_).squeeze()
        if self.sent_size == self.output_size:
                    asp_proj = attn_applied_.unsqueeze(1)
        else:
            asp_proj = self.dimproj(attn_applied_).unsqueeze(1)
        #print "Output size,", output.size()
        #print "Aspect proj size,", asp_proj.size()

        output=output.permute(0,2,1)

        betas = []
        for i in range(0,self.hop_size):
            match = torch.bmm(asp_proj,output).permute(2,0,1)


            attn_weights2 = F.softmax(
                    self.attn2(match), dim=0)
            #print attn_weights
            self.hidden_aspect_write_gru=self.init_hidden(batch_size)
            output_w, hidden_aspect_write_gru = \
            self.aspect_write_gru(output.permute(2,0,1),self.hidden_aspect_write_gru)

            output_w = torch.mul(output_w.permute(1,0,2),temp_attention_mat2)
            output_w = self.dropout2(output_w)


            masked_attn_weights2 = torch.mul(attn_weights2.squeeze().permute(1,0),attention_mat2)
            #print masked_attn_weights.size()
            _sums2 = masked_attn_weights2.sum(-1).unsqueeze(1).expand(-1,masked_attn_weights2.size()[1])
            #print _sums.size()
            attentions2 = masked_attn_weights2.div(_sums2).unsqueeze(1).permute(2,0,1)

            #print output_w.size()
            #print attn_weights.size()

            #print attentions2.squeeze().permute(1,0)[0].sum()

            attn_applied = torch.bmm(attentions2.permute(1,2,0), output_w.permute(0,1,2)).squeeze()

            betas.append(attentions2.permute(1,2,0))

            #print "attn_applied size", attn_applied.size()

            query = asp_proj.view(asp_proj.size()[0],asp_proj.size()[2])

            #print "query size", query.size()

            final_output = torch.add(attn_applied, query)

            #print final_output.size()

            final_output = F.relu(final_output)
            asp_proj = final_output.unsqueeze(1)
            #output = output_w.permute(1,2,0)
            output = output_w.permute(0,2,1)
            #print"output size final-----", output.size()
        asp_proj = F.log_softmax(self.affine(asp_proj.squeeze()),dim=1)
        #asp_proj = self.affine(asp_proj.squeeze())
        return asp_proj, betas, torch.cat(alphas,0)

    def init_hidden(self, batch_size):

        return autograd.Variable(torch.zeros(1, batch_size,
            self.output_size).type(ftype))

    def init_hidden_memnet(self, batch_size):

        return autograd.Variable(torch.zeros(2, batch_size,
            self.output_size/2).type(ftype))

   # def init_aspect_hidden(self, batch_size):
   #    return autograd.Variable(torch.zeros(1, batch_size, self.sent_size))

    def init_hidden2(self, batch_size):

        return autograd.Variable(torch.zeros(1, batch_size,
            self.sent_size).type(ftype))

def Glove(GLOVE_DIR):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.840B.300d.txt'))
    #f = open(os.path.join(GLOVE_DIR, 'ex.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

def index_word_embeddings(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)


def train(onea):
    tokenizer = Tokenizer(num_words=nb_words)
    prep = PreProcessing(training_data,test_data,tokenizer,BATCH_SIZE)
    data = prep.keras_data_prepare()

    we=Glove(GLOVE_DIR="/home/navonil/")
    ei=index_word_embeddings(tokenizer.word_index,we)
    word_embeddings = nn.Embedding(len(tokenizer.word_index)+1, EMBEDDING_DIM,padding_idx=0)
    word_embeddings.weight = nn.Parameter(torch.FloatTensor(ei) if not args.cuda else torch.cuda.FloatTensor(ei))
    word_embeddings.weight.requires_grad = False
    print "Embeddings loaded...."

    model = AttnRNN(HOP_SIZE, BATCH_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    if args.cuda:
        model.cuda()
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, [x for x in
        model.parameters()] + [word_embeddings.weight]), lr = args.lr,
        weight_decay = args.l2)

    batch_count = int(np.ceil(len(training_data)/float(BATCH_SIZE)))

    for i in range(NB_EPOCH):
        start_time = time.time()

        loss_tot = []
        true_label=[]
        pred_res=[]
        model.train()
        for batch_id in range(batch_count):
            optimizer.zero_grad()
            bdata_x0, bdata_x1, bdata_y, attention_mat2, attention_mat1 = prep.prepare_data(data, batch_id, word_embeddings)
            model.hidden_sentence_gru = model.init_hidden2(bdata_x0.size()[0])
            model.hidden_aspect_gru = model.init_hidden(bdata_x0.size()[0])
            model.hidden_aspect_write_gru = model.init_hidden(bdata_x0.size()[0])
            #model.hidden_aspect_repr_gru = model.init_aspect_hidden(bdata_x0.size()[0])


            prediction, _, _ = model(bdata_x0,bdata_x1, attention_mat1, attention_mat2, bdata_x0.size()[0])
            loss = loss_function(prediction, torch.max(bdata_y, 1)[1])
            # print "Loss ", i, loss.data[0]
            loss_tot.append(loss.data[0])
            pred_label = prediction.data.max(1)[1].cpu().numpy()
            pred_res += [x for x in pred_label]
            true_data = torch.max(bdata_y, 1)[1].cpu()
            true_label+= [x for x in true_data.data]
            loss.backward()
            # print word_embeddings.weight.grad
            optimizer.step()

        preds,true,test_loss = test(test_data, model, tokenizer,
                word_embeddings, loss_function, i,onea)

        # for k in range(1,39):
        #     print '%s, %s, %d, %d' % (test_data[-k][0],test_data[-k][2],true[-k],preds[-k])

        print 'Epoch %d train_loss %.4f train_acc %.2f test_loss %.4f test_acc %.2f time %.2f' % (i+1, np.mean(loss_tot), accuracy(pred_res, true_label), test_loss, accuracy(preds,true), time.time()-start_time)
        # import ipdb;ipdb.set_trace()
        mul = set(range(len(true)))-set(onea)
        print 'single_aspect %.2f mul_aspect %.2f' % (accuracy([preds[idx] for idx in onea],[true[idx] for idx in onea]), accuracy([preds[idx] for idx in mul],[true[idx] for idx in mul]))

    return model, tokenizer, word_embeddings


def test(test_data, model, tokenizer, word_embeddings, loss_function, epoch, onea):
    prep = PreProcessing(test_data,training_data,tokenizer,BATCH_SIZE)
    data = prep.keras_data_prepare(False)

    model.eval()
    true_label=[]
    loss_tot = []
    pred_res=[]

    batch_count = int(np.ceil(len(test_data)/float(BATCH_SIZE)))

    # print batch_count, len(test_data)
    betas = []
    alphas = []
    for batch_id in range(batch_count):
            bdata_x0, bdata_x1, bdata_y, attention_mat2, attention_mat1 = prep.prepare_data(data, batch_id, word_embeddings)
            model.hidden_sentence_gru = model.init_hidden2(bdata_x0.size()[0])
            model.hidden_aspect_gru = model.init_hidden(bdata_x0.size()[0])
            model.hidden_aspect_write_gru = model.init_hidden(bdata_x0.size()[0])
            #model.hidden_aspect_repr_gru = model.init_aspect_hidden(bdata_x0.size()[0])

            preds, beta , alpha = model(bdata_x0,bdata_x1, attention_mat1, attention_mat2, bdata_x0.size()[0])
            betas +=[dat.data.cpu().numpy() for dat in beta]
            alphas.append(alpha.data.cpu().numpy())
            loss = loss_function(preds, torch.max(bdata_y, 1)[1])
            loss_tot.append(loss.data[0])
            pred_label = preds.data.max(1)[1].cpu().numpy()
            pred_res += [x for x in pred_label]
            true_data = torch.max(bdata_y, 1)[1].cpu()
            true_label+= [x for x in true_data.data]
    # with open('betas_%d.p'%epoch,'wb') as fp:
    #     cPickle.dump(betas,fp)
    # with open('alphas_%d.p'%epoch,'wb') as fp:
    #     cPickle.dump(alphas,fp)
    return pred_res, true_label, np.mean(loss_tot)

def csv_reader(file):
    data =[]
    with open(file, 'rb') as csvfile:
        aspectreader = csv.reader(csvfile, delimiter=',')
        for row in aspectreader:
            sent = row[0].lower()
            nb_aspects = int(row[1])
            aspects = [x.replace("'","").replace('[',"").replace("\"","").replace(']',"").strip().lower() for x in row[2].split(",")]
            sentiments = [x.strip().replace("'","").replace('[',"").replace("\"","").replace(']',"").lower() for x in row[3].split(",")]
            for i in range(0,nb_aspects):
                datam = (sent,aspects , aspects[i], [sentiments[i]])
                data.append(datam)
    return data

def accuracy(preds, true):
    return sum(1 for x,y in zip(preds,true) if x == y) / float(len(preds))*100.

if __name__=='__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='does not use GPU')
    # parser.add_argument('--dataset', default='Laptop', metavar='D',
    #                     help='Laptop or Restaurants')
    # args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print 'Running on GPU'
        torch.cuda.manual_seed(1)
        ftype = torch.cuda.FloatTensor
    else:
        print 'Running on CPU'
        torch.manual_seed(1)
        ftype = torch.FloatTensor
    training_data = csv_reader('2014_'+args.dataset+'_train.csv')
    test_data = csv_reader('2014_'+args.dataset+'_test.csv')
    shuffle(training_data)
    # print training_data[0]
    # print np.max([len(x.split()) for x in zip(*training_data)[0]+zip(*test_data)[0]])
    # print np.max([len(x.split()) for x in zip(*training_data)[2]+zip(*test_data)[2]])
    # print np.max([len(x) for x in zip(*training_data)[1]+zip(*test_data)[1]])
    # sys.exit(0)

    onea = [i for i,(s,a,aa,l) in enumerate(test_data) if len(a)==1]
    tonea = [i for i,(s,a,aa,l) in enumerate(training_data) if len(a)==1]
    print len(onea),len(test_data)-len(onea)
    print len(tonea),len(training_data)-len(tonea)
    model, tokenizer, word_embeddings = train(onea)
