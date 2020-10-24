import torch
from torch.utils.data import Dataset
from scipy.misc import imread, imresize
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, structure_word_map, cell_word_map, max_structure_size,
                 max_cell_size, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        :param structure_word_map: structure word map
        :param cell_word_map: cell word map
        """
        self.max_structure_size = max_structure_size
        self.max_cell_size = max_cell_size
        self.all_data = []
        self.structure_word_map = structure_word_map
        self.cell_word_map = cell_word_map
        data_set = os.path.join(data_folder, 'PubTabBet_2.0.0.jsonl')
        with open(data_set, 'r') as f:
            for line in f:
                record = json.loads(line)
                type = record['split']
                if type == 'train':
                    train_dir = os.path.join(data_folder, type)
                    data['img'] = os.path.join(train_dir, record['filename'])
                    data['structure'] = record['html']['structure']
                    data['cells'] = record['html']['cells']
                    data['split'] = record['split']
                    self.all_data.append(data)

        # self.split = split
        #
        # assert self.split in {'train', 'val', 'test'}

        # Open hdf5 file where images are stored
        # self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        # self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.all_data)

    @staticmethod
    def truncate_or_padding(tokens, word_map, max_length):
        r = ['<start>', tokens, '<end>']
        idx_rep = [word_map[i] for i in r]
        result_tensor = torch.ones(word_map['<pad>']) * max_length
        if len(idx_rep) <= len(result_tensor):
            result_tensor[:len(idx_rep) + 1] = torch.tensor(idx_rep)
        else:
            truncated = idx_rep[:-1].append(word_map['<end>'])
            result_tensor = torch.tensor(idx_rep[:-1].append(truncated))
        return result_tensor

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.all_data[i]['img']
        # if self.transform is not None:
        #     img = self.transform(img)

        # convert the caption sentences to vector by word map
        with open(self.structure_word_map, 'r') as f:
            s = json.load(f)
        with open(self.cell_word_map, 'r') as f:
            c = json.load(f)

        structure_caption = truncate_or_padding(self.all_data[i]['structure'], s, self.max_structure_size)
        cell_caption = None
        cell_length = []
        for cell in self.all_data[i]['cells']:
            if cell_caption is None:
                cell_caption = torch.tensor(truncate_or_padding(cell['tokens'], c,  self.max_cell_size))
                cell_length.append(len(cell['token']))
            else:
                row_caption = torch.tensor(truncate_or_padding(cell['tokens'], c, self.max_cell_size))
                cell_caption = torch.vstack(cell_caption, row_caption)


        # content = ['<start>']
        # content.extend(self.all_data[i]['structure'])
        # content.append('<end>')
        # structure_caption = ([s[x] for x in content])
        # structure_caption = torch.zeros(self.max_structure_size)
        # structure_caption = torch.LongTensor(structure_caption)
        # if len(structure_caption) <= self.max_structure_size:
        #     # padding
        #     structure_caption[len(structure_caption):] = s['<pad>']
        # else:
        #     structure_caption[-1] = s['<end>']

        # cells_caption = torch.LongTensor(self.all_data[i]['cells'])
        structure_caption_length = torch.LongTensor([len(self.all_data[i]['structure'])])

        cell_caption_length = torch.LongTensor
        # cells_caption_length = torch.LongTensor([len(self.all_data[i]['cell'])])

        # caplen = torch.LongTensor([self.caplens[i]])

        if self.all_data[i]['split'] is 'TRAIN':
            return img, structure_caption, cells_caption, structure_caption_length, cells_caption_length
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return len(self.all_data)
