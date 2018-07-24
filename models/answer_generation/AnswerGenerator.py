import random

import time
import torch

import constants as c
from models.answer_generation import MAX_LENGTH, DROPOUT, LSTM_SIZE, EPOCHS
from models.answer_generation.nn_model import EncoderRNN, AttnDecoderRNN
from utils.dependency_parser import get_spacy_parser
from torch.autograd import Variable

import utils.misc as misc
import utils.data as datautils

SOS_token = 0
EOS_token = 1

teacher_forcing_ratio = 0.5
hidden_size = LSTM_SIZE

class AnswerGenerator():
    def __init__(self):
        self.word2index = {"SOS":0, "EOS":1, c.UNK_TAG: 2}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: c.UNK_TAG}
        self.n_words = 2  # Count SOS and EOS

        self.use_cuda = torch.cuda.is_available()
        self.SOS_token = 0
        self.EOS_token = 1

        self.encoder = None
        self.decoder = None


    def save(self, dirpath):
        torch.save(self.encoder, dirpath+"/encoder")
        torch.save(self.decoder, dirpath + "/decoder")
        self.encoder = None
        self.decoder = None
        import pickle
        pickle.dump(self, open(dirpath+"/model", "wb"))
        return True

    @staticmethod
    def load(dirpath):
        import pickle
        ag = pickle.load(open(dirpath+"/model", "rb"))
        ag.encoder = torch.load(dirpath+"/encoder")
        ag.decoder = torch.load(dirpath+"/decoder")
        return ag



    def addSentence(self, sentence):
        parser = get_spacy_parser()
        for word in parser.tokenizer(sentence):
            self.addWord(word.text)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def indexesFromSentence(self, sentence):
        return [self.word2index[word] if word in self.word2index else self.word2index[c.UNK_TAG] for word in
                sentence.split(' ')]

    def variableFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        indexes.append(EOS_token)
        result = Variable(torch.LongTensor(indexes).view(-1, 1))
        if self.use_cuda:
            return result.cuda()
        else:
            return result

    def variablesFromPair(self, pair):
        input_variable = self.variableFromSentence(pair[0])
        target_variable = self.variableFromSentence(pair[1])
        return (input_variable, target_variable)

    def filterPair(self, p):
        parser = get_spacy_parser()
        return len(parser.tokenizer(p[0])) < MAX_LENGTH and \
               len(parser.tokenizer(p[1])) < MAX_LENGTH

    def filterPairs(self, pairs):
        return [pair for pair in pairs if self.filterPair(pair)]

    def normalize_string(self, string):
        return datautils.clean_question(string)


    def normalize_pairs(self, pairs):
        return [(self.normalize_string(p[0]), self.normalize_string(p[1])) for p in pairs]

    def make_training_pairs(self, X,Y):
        pairs = list(zip(X, Y))
        pairs = self.filterPairs(pairs)
        pairs = self.normalize_pairs(pairs)

        return pairs

    def train(self, X, Y):

        pairs = self.make_training_pairs(X, Y)

        for pair in pairs:
            self.addSentence(pair[0])
            self.addSentence(pair[1])

        self.encoder = EncoderRNN(len(self.index2word), hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size, len(self.index2word), 1, dropout_p=DROPOUT)
        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.trainIters(pairs,  print_every=10)

    def trainIters(self, pairs, epochs=EPOCHS, print_every=1000, learning_rate=0.01):
        start = time.time()

        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=learning_rate)

        training_pairs = [self.variablesFromPair(p) for p in pairs]

        criterion = torch.nn.NLLLoss()

        n_iters = len(training_pairs)
        for e in range(epochs):
            print("Epoch %d" %e)
            for iter in range(1, n_iters+1):
                training_pair = training_pairs[iter - 1]

                input_variable = training_pair[0]
                target_variable = training_pair[1]

                loss = self._train(input_variable, target_variable, self.encoder,
                             self.decoder, encoder_optimizer, decoder_optimizer, criterion)
                print_loss_total += loss
                plot_loss_total += loss

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (misc.timeSince(start, iter / n_iters),
                                                 iter, iter / n_iters * 100, print_loss_avg))



    def _train(self, input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
              max_length=MAX_LENGTH):
        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])
                decoder_input = target_variable[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_output, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                ni = topi[0][0]

                decoder_input = Variable(torch.LongTensor([[ni]]))
                decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

                loss += criterion(decoder_output, target_variable[di])
                if ni == EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / target_length

    def evaluate(self, sentence, max_length=MAX_LENGTH):
        new_sentence = self.normalize_string(sentence)

        input_variable = self.variableFromSentence(new_sentence)
        input_length = input_variable.size()[0]
        encoder_hidden = self.encoder.initHidden()

        encoder_outputs = Variable(torch.zeros(max_length, self.encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_output, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        return decoded_words, decoder_attentions[:di + 1]


    def evaluate_pair(self, pair):
        output_words = self.evaluate(pair[0])
        output_sentence = ' '.join(output_words)
        return output_sentence
