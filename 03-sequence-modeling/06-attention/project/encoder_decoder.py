import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import en_core_web_sm

# Load the spaCy English tokenizer
spacy_en = en_core_web_sm.load()

# Tokenize a sentence using spaCy


def tokenize(text):
    return [token.text.lower() for token in spacy_en.tokenizer(text)]

# Create a small vocabulary


class Vocab:
    def __init__(self):
        self.word2index = {"<SOS>": 0, "<EOS>": 1, "<PAD>": 2, "<UNK>": 3}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2: "<PAD>", 3: "<UNK>"}
        self.n_words = 4
        self.word_counts = {}

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word_counts[word] = 1
            self.n_words += 1
        else:
            self.word_counts[word] += 1

# Encoder


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Decoder


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

# Training


def train(encoder, decoder, input_tensor, target_tensor, optimizer, criterion, vocab, max_length):
    encoder_hidden = encoder.init_hidden()

    optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[vocab.word2index["<SOS>"]]])
    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, _ = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()

        loss += criterion(decoder_output,
                          target_tensor[di])
        if decoder_input.item() == vocab.word2index["<EOS>"]:
            break

    loss.backward()

    optimizer.step()

    return loss.item() / target_length

# Evaluation


def evaluate(encoder, decoder, sentence, vocab, max_length):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(sentence, vocab)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[vocab.word2index["<SOS>"]]])
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == vocab.word2index["<EOS>"]:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

# Convert a sentence to a tensor


def tensor_from_sentence(sentence, vocab):
    indexes = [vocab.word2index[word]
               if word in vocab.word2index else vocab.word2index["<UNK>"] for word in sentence]
    indexes.append(vocab.word2index["<EOS>"])
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


# def tensor_from_sentence(sentence, vocab):
#     indexes = [vocab.word2index[word]
#                if word in vocab.word2index else vocab.word2index["<UNK>"] for word in sentence]
#     indexes.append(vocab.word2index["<EOS>"])
#     one_hot_vectors = torch.zeros(
#         (len(indexes), vocab.n_words), dtype=torch.long)
#     for i, index in enumerate(indexes):
#         one_hot_vectors[i][index] = 1
#     return one_hot_vectors
# Convert a tensor to a sentence


def sentence_from_tensor(tensor, vocab):
    return [vocab.index2word[index.item()] for index in tensor]

# Main function


def main():
    # Define your input and output sizes, hidden size, and other hyperparameters
    hidden_size = 256
    learning_rate = 0.01
    num_iterations = 1000

    # Create a vocabulary and add the input and target sentences
    vocab = Vocab()
    input_sentence = tokenize("the quick brown fox jumps over the lazy dog")
    target_sentence = tokenize("the quick brown fox jumps over the lazy dog")
    vocab.add_sentence(input_sentence)
    vocab.add_sentence(target_sentence)

    # Create instances of the encoder and decoder
    encoder = Encoder(vocab.n_words, hidden_size)
    decoder = Decoder(hidden_size, vocab.n_words)

    # Define the optimizer and loss function
    optimizer = optim.SGD(list(encoder.parameters()) +
                          list(decoder.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Convert input and target sentences to tensors
    input_tensor = tensor_from_sentence(input_sentence, vocab)
    target_tensor = tensor_from_sentence(target_sentence, vocab)

    # Training loop
    for iteration in range(num_iterations):
        loss = train(encoder, decoder, input_tensor, target_tensor,
                     optimizer, criterion, vocab, max_length=len(input_tensor))
        print(f"Iteration: {iteration+1}, Loss: {loss:.4f}")

    # Evaluate the model
    input_sentence = tokenize("the quick brown fox")
    decoded_words = evaluate(
        encoder, decoder, input_sentence, vocab, max_length=10)
    output_sentence = ' '.join(decoded_words)
    print(f"Input: {' '.join(input_sentence)}\nOutput: {output_sentence}")


if __name__ == '__main__':
    main()
