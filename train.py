import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import os
import json
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, StructureDecoderWithAttention, CellDecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from scipy.misc import imread, imresize

# Data parameters
data_folder = '/home2/lihuichao/dataset/pubtabnet/'  # folder with data files saved by create_input_files.py
# data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
# device = torch.device("cpu")  # sets device for model and PyTorch tensors

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 5  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 1
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-5  # learning rate for encoder if fine-tuning
decoder_lr = 4e-5  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
checkpoint = None  # path to checkpoint, None if none
para_lambda = 0.6


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map, td_symbol, bracket_symbol

    # Read word map
    with open('structure_text.json', 'r') as j:
        structure_word_map = json.load(j)
    with open('cell_text.json', 'r') as f:
        cell_word_map = json.load(f)
    td_symbol = structure_word_map['<td>']
    bracket_symbol = structure_word_map['>']

    # Initialize / load checkpoint
    if checkpoint is None:
        structure_decoder = StructureDecoderWithAttention(attention_dim=attention_dim,
                                                          embed_dim=emb_dim,
                                                          decoder_dim=decoder_dim,
                                                          vocab_size=len(structure_word_map),
                                                          dropout=dropout)
        cell_decoder = CellDecoderWithAttention(attention_dim=attention_dim,
                                                embed_dim=emb_dim,
                                                decoder_dim=decoder_dim,
                                                vocab_size=len(cell_word_map),
                                                dropout=dropout)
        encoder = Encoder()
        decoder_optimizer = torch.optim.Adam(
            [
                {'params': filter(lambda p: p.requires_grad, structure_decoder.parameters())},
                {'params': filter(lambda p: p.requires_grad, cell_decoder.parameters())},
            ],
            lr=decoder_lr)
        # encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr)

    # else:
    #     checkpoint = torch.load(checkpoint)
    #     start_epoch = checkpoint['epoch'] + 1
    #     epochs_since_improvement = checkpoint['epochs_since_improvement']
    #     best_bleu4 = checkpoint['bleu-4']
    #     decoder = checkpoint['decoder']
    #     decoder_optimizer = checkpoint['decoder_optimizer']
    #     encoder = checkpoint['encoder']
    #     encoder_optimizer = checkpoint['encoder_optimizer']
    #     if fine_tune_encoder is True and encoder_optimizer is None:
    #         encoder.fine_tune(fine_tune_encoder)
    #         encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
    #                                              lr=encoder_lr)

    # Move to GPU, if available
    encoder = encoder.to(device)
    structure_decoder = structure_decoder.to(device)
    cell_decoder = cell_decoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, structure_word_map, cell_word_map, 512, 512),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, structure_word_map, cell_word_map, 512, 512),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              structure_decoder=structure_decoder,
              cell_decoder=cell_decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        # recent_bleu4 = validate(val_loader=val_loader,
        #                         encoder=encoder,
        #                         structure_decoder=decoder,
        #                         cell_decoder=cell_decoder,
        #                         criterion=criterion)
        #
        # # Check if there was an improvement
        # is_best = recent_bleu4 > best_bleu4
        # best_bleu4 = max(recent_bleu4, best_bleu4)
        # if not is_best:
        #     epochs_since_improvement += 1
        #     print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        # else:
        #     epochs_since_improvement = 0

        # Save checkpoint
        # save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
        #                 decoder_optimizer, recent_bleu4, is_best)
        if os.path.exists('./model.pth'):  # checking if there is a file with this name
            os.remove('./model.pth')  # deleting the file
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'structure_decoder_state_dict': structure_decoder.state_dict(),
            'cell_decoder_state_dict': cell_decoder.state_dict(),
            'optimizer_state_dict': decoder_optimizer.state_dict(),
        }, './model.pth')


def read_image_from_path(im_path):
    img = imread(im_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).unsqueeze(0)
    return img


def single_record_loss(output, target, length, criterion, alphas):
    target = target[:, 1:]
    # output = output[:, 1:]
    scores = pack_padded_sequence(output, length, batch_first=True)
    targets = pack_padded_sequence(target, length, batch_first=True)
    # print(scores.data.shape, targets.data.shape)
    loss = criterion(scores.data, targets.data)
    loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
    return loss, targets.data


def train(train_loader, encoder, structure_decoder, cell_decoder, criterion, encoder_optimizer, decoder_optimizer,
          epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param structure_decoder: structure decoder model
    :param cell_decoder: cell decoder
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    structure_decoder.train()  # train mode (dropout and batchnorm is used)
    cell_decoder.train()
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    structure_losses = AverageMeter()  # loss (per word decoded)
    cell_losses = AverageMeter()
    total_losses = AverageMeter()
    structure_top5accs = AverageMeter()  # top5 accuracy
    cell_top5accs = AverageMeter()
    total_losses = AverageMeter()

    start = time.time()

    # Batches
    for i, (img_path, structure_caption, cell_caption, structure_caption_length, cell_caption_length) in enumerate(
            train_loader):
        # print(structure_caption, img_path)
        data_time.update(time.time() - start)
        img_path = img_path[0]
        # Move to GPU, if available
        img = read_image_from_path(img_path)
        img = img.to(device)
        structure_caption, cell_caption = structure_caption.to(device), cell_caption.to(device)
        structure_caption_length, cell_caption_length = structure_caption_length.to(device), cell_caption_length.to(
            device)
        # caps = caps.to(device)
        # caplens = caplens.to(device)

        # Forward prop.
        feature = encoder(img)
        structure_scores, structure_caps_sorted, structure_decode_lengths, structure_alphas, structure_sort_ind, hidden = structure_decoder(
            feature, structure_caption, structure_caption_length)
        # print(structure_scores.shape, structure_caps_sorted.shape, structure_decode_lengths, structure_alphas.shape,
        #       structure_sort_ind.shape)
        structure_loss, structure_target = single_record_loss(structure_scores, structure_caps_sorted,
                                                              structure_decode_lengths,
                                                              criterion, structure_alphas)
        # 使用teacher forcing，当structure_cas_sorted的结果为<td>或>时，对cell进行预测
        cell_idx = torch.logical_or((structure_caps_sorted == td_symbol), (structure_caps_sorted == bracket_symbol))[:,
                   :structure_decode_lengths[0]]
        structure_hidden = hidden[cell_idx].unsqueeze(0)  # (batch_size, decoder_dim)
        # 找出根据encoder_out和hidden状态进行解码
        cell_loss = None
        for j in range(cell_caption.size(1)):
            cell_scores, cell_caps_sorted, cell_decode_lengths, cell_alphas, cell_sort_ind = cell_decoder(
                feature, cell_caption[:, j, :], cell_caption_length[:, j, :], structure_hidden[:, j, :])
            # print(f'shape: {cell_decode_lengths.shape}\t value: {cell_decode_lengths}')
            loss, _ = single_record_loss(cell_scores, cell_caps_sorted, cell_decode_lengths, criterion, cell_alphas)
            cell_losses.update(loss.item(), sum(cell_decode_lengths))
            if cell_loss:
                cell_loss = cell_loss + loss
            else:
                print(f'length of cell decoder: {cell_decode_lengths}')
                cell_loss = loss
            print(f'Cell Loss per cell {cell_losses.val:.4f} ({cell_losses.avg:.4f})\t'.format(cell_losses=cell_losses))
        total_loss = para_lambda * structure_loss + (1 - para_lambda) * cell_loss
        print(f'cell loss: {cell_loss}\ttotal_loss: {total_loss}')
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this

        # Calculate loss
        # print(structure_scores)

        # Add doubly stochastic attention regularization

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics

        # structure_top5 = accuracy(structure_scores, structure_target, 5)
        structure_losses.update(structure_loss.item(), sum(structure_decode_lengths))
        # structure_top5accs.update(structure_top5, sum(structure_decode_lengths))

        # cell_top5 = accuracy(cell_scores, cell_caps_sorted[:, 1:], 5)
        # cell_losses.update(cell_loss.item(), sum(cell_decode_lengths))
        # cell_top5accs.update(cell_top5, sum(cell_decode_lengths))

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Structure Loss {structure_loss.val:.4f} ({structure_loss.avg:.4f})\t'.format(epoch, i,
                                                                                                len(train_loader),
                                                                                                batch_time=batch_time,
                                                                                                data_time=data_time,
                                                                                                structure_loss=structure_losses))
            # structure_top5=structure_top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    try:
     main()
    except Exception as e:
        print(e)
