import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, structure_word_map, cell_word_map, max_structure_size, max_cell_size):
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
        data_set = os.path.join(data_folder, 'PubTabNet_2.0.0.jsonl')
        with open(data_set, 'r') as f:
            for line in f:
                record = json.loads(line)
                type = record['split']
                if type == 'train':
                    train_dir = os.path.join(data_folder, type)
                    data = {'img': os.path.join(train_dir, record['filename']),
                            'structure': record['html']['structure'], 'cells': record['html']['cells'],
                            'split': record['split']}
                    self.all_data.append(data)

        # self.split = split
        #
        # assert self.split in {'train', 'val', 'test'}

        # Open hdf5 file where images are stored
        # self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        # self.imgs = self.h['images']

        # Captions per image
        # self.cpi = self.h.attrs['captions_per_image']
        #
        # # Load encoded captions (completely into memory)
        # with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
        #     self.captions = json.load(j)
        #
        # # Load caption lengths (completely into memory)
        # with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
        #     self.caplens = json.load(j)
        #
        # # PyTorch transformation pipeline for the image (normalizing, etc.)
        # self.transform = transform
        #
        # # Total number of datapoints
        # self.dataset_size = len(self.all_data)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.all_data[i]['img']
        # if self.transform is not None:
        #     img = self.transform(img)

        # convert the caption sentences to vector by word map

        structure_caption = truncate_or_padding(self.all_data[i]['structure']['tokens'], self.structure_word_map,
                                                self.max_structure_size)
        structure_caption = torch.tensor(structure_caption)
        cell_caption = []
        cell_caption_length = []
        for cell in self.all_data[i]['cells']:
            # cell_caption = torch.tensor(truncate_or_padding(cell['tokens'], self.cell_word_map,
            # self.max_cell_size)).unsqueeze(0)
            c = len(cell['tokens']) + 2
            c = c if c <= self.max_cell_size else self.max_cell_size
            cell_caption.append(truncate_or_padding(cell['tokens'], self.cell_word_map, self.max_cell_size))
            cell_caption_length.append([c])
        cell_caption = torch.tensor(cell_caption)
        cell_caption_length = torch.tensor(cell_caption_length)

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

        # ！需要判断这里指的length包不包括<start>和<end>
        s = len(self.all_data[i]['structure']['token']) + 2
        s = s if s <= self.max_structure_size else self.max_structure_size
        structure_caption_length = torch.tensor([s])
        # two dimension vector
        # todo: 处理长度溢出的情况

        # cells_caption_length = torch.LongTensor([len(self.all_data[i]['cell'])])
        # caplen = torch.LongTensor([self.caplens[i]])

        return img, structure_caption, cell_caption, structure_caption_length, cell_caption_length
        # else:
        # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
        # todo: 测试集返回数据
        # all_captions = torch.LongTensor(
        #     self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
        # return img, caption, caplen, all_captions
        # pass

    def __len__(self):
        return len(self.all_data)


def truncate_or_padding(tokens, word_map, max_length):
    r = ['<start>']
    r.extend(tokens)
    r.append('<end>')
    idx_rep = [word_map[i] for i in r]
    result_tensor = [word_map['<pad>']] * max_length
    if len(idx_rep) <= len(result_tensor):
        result_tensor[:len(idx_rep)] = idx_rep
    else:
        truncated = idx_rep[:max_length-1]
        truncated.append(word_map['<end>'])
        result_tensor = truncated
    return result_tensor


if __name__ == '__main__':
    tokens = ['<start>'] * 600
    word_map = {'<start>': 0, '<pad>': 2, '<end>': 3}
    max_length = 512
    result = truncate_or_padding(tokens, word_map, max_length)
    print(f'result: {result}\tresult length: {len(result)}')
