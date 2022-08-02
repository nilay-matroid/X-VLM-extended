import json
import os

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import PIL
from PIL import Image
from random import randint, choice

""" Real Scenario Text-based Person Re-identifications dataset.

    Reference:
    RSTPReid: Large-Scale Spatio-Temporal Person Re-identification
    URL: https://github.com/njtechcvlab/rstpreid-dataset

    Google Drive
    Link: https://drive.google.com/file/d/1CQqDcquTq17FJ84IvXtzdxTqf6FyPSQv/view?usp=sharing

    Dataset statistics:
        To properly handle real scenarios, we construct a new dataset called Real Scenario Text-based Person Re-identification (RSTPReid) 
        based on MSMT17 [2]. RSTPReid contains 20505 images of 4,101 persons from 15 cameras. 
        Each person has 5 corresponding images taken by different cameras with complex both indoor and outdoor scene 
        transformations and backgrounds in various periods of time, which makes RSTPReid much more challenging and 
        more adaptable to real scenarios. Each image is annotated with 2 textual descriptions. For data division, 3701 (index < 18505), 
        200 (18505 <= index < 19505) and 200 (index >= 19505) identities are utilized for training, validation and testing, respectively 
        (Marked by item 'split' in the json file). Each sentence is no shorter than 23 words.
        
        Original Dataset (unfiltered)
        Train set: 3701 identities, 18505 images
        Validation set:  200 identities, 1000 images
        Test setand 200 identities, 1000 images
        --------------------------------------
        subset         | # ids     | # images
        --------------------------------------
        train          |   3701    |    18505
        valid          |    200    |     1000
        test           |    200    |     1000

        -----------------------------------------------------------------
        -----------------------------------------------------------------
        
        Filtered Dataset
        Train set: 3645 identities, 17467 images
        Validation set:  192 identities, 878 images
        Test setand 199 identities, 966 images
        --------------------------------------
        subset         | # ids     | # images
        --------------------------------------
        train          |   3645    |    17467
        valid          |    192    |     878
        test           |    199    |     966
"""

valid_splits = {'train', 'val', 'test', 'all'}


class RSTPReidTrain(Dataset):
    def _validate_args(self):
        assert os.path.isdir(self.root_dir), f"The root directory for RSTPReid {self.root_dir} doesn't exist!"
        assert os.path.isdir(self.image_file_path), f"The image directory for RSTPReid {self.image_file_path} doesn't exist!"
        assert os.path.isfile(self.captions_file), f"The captions file for RSTPReid {self.captions_file} doesn't exist!"
        assert self.split in valid_splits, f"Invalid split name: {self.split}"
        

    def __init__(self, root_dir, image_transform, split='train', shuffle=False, print_stats=True):
        """
        Args:
            root_dir (string): Directory with all the images and captions.
        """
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.image_file_path = os.path.join(root_dir, 'imgs')
        self.captions_file = os.path.join(root_dir, 'filtered_data_captions.json')
        self.shuffle = shuffle
        self.image_transform = image_transform
        self._validate_args()

        dataframe = pd.read_json(self.captions_file)
        if print_stats:
            self._print_dataset_stats(dataframe)

        if split != 'all':
            dataframe = dataframe[dataframe['split'] == split]

        self.imgs = dataframe['img_path'].tolist()
        self.identities = dataframe['id'].tolist()
        self.captions = dataframe['captions'].tolist()
        self.unique_identities = dataframe['id'].unique().tolist()

        self.id2img = {}
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id in range(len(self.captions)):
            # Add image id
            id = self.identities[img_id]
            if id not in self.id2img:
                self.id2img[id] = []
            self.id2img[id].append(img_id)

            # Add text id
            self.img2txt[img_id] = []
            for _ in self.captions[img_id]:
                self.txt2img[txt_id] = img_id
                self.img2txt[img_id].append(txt_id)
                txt_id += 1
            
    def __len__(self):
        return len(self.unique_identities)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        identity = self.unique_identities[idx]

        # Randomly choose image id for the corresponding identity. (choose 1 out of 5)
        try:
            image_id = choice(self.id2img[identity])
        except IndexError as zero_imageids_ex:
            print(f"An exception occurred trying to select image id for identity {identity}.")
            print(f"Skipping index {idx}")
            return self.skip_sample(idx)

        # Randomly choose caption for the corresponding image id. (choose 1 out of 2)
        try:
            caption = choice(self.captions[image_id])
        except IndexError as zero_textids_ex:
            print(f"An exception occurred trying to select text id for identity {identity} and image id {image_id}.")
            print(f"Skipping index {idx}")
            return self.skip_sample(idx)

        # Load Image Tensor
        image_file = os.path.join(self.image_file_path, self.imgs[image_id])
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {idx}")
            return self.skip_sample(idx)

        return image_tensor, caption, image_id

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def _print_dataset_stats(self, df):
        train = df[df['split'] == 'train']
        val = df[df['split'] == 'val']
        test = df[df['split'] == 'test']

        print("--------------------------------------")
        print("subset         | # ids     | # images")
        print("--------------------------------------")
        print(f"train          |   {len(train['id'].unique())}    |    {len(train)}")
        print(f"valid          |    {len(val['id'].unique())}    |     {len(val)}")
        print(f"test           |    {len(test['id'].unique())}    |     {len(test)}")


class RSTPReid(Dataset):
    def __init__(self, root_dir, split='train', transform=None, print_stats=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert os.path.isdir(root_dir), f"The root directory for RSTPReid {root_dir} doesn't exist!"
        self.image_file_path = os.path.join(root_dir, 'imgs')
        assert os.path.isdir(self.image_file_path), f"The image directory for RSTPReid {self.image_file_path} doesn't exist!"
        
        assert split in valid_splits, f"Invalid split name: {split}"

        self.caption_dir = os.path.join(root_dir, 'filtered_data_captions.json')
        assert os.path.isfile(self.caption_dir), f"The caption file for RSTPReid {self.caption_dir} doesn't exist!"
        dataframe = pd.read_json(self.caption_dir)

        if print_stats:
            self._print_dataset_stats(dataframe)

        if split != 'all':
            dataframe = dataframe[dataframe['split'] == split]

        self.split = split
        self.imgs = dataframe['img_path'].tolist()
        self.identities = dataframe['id'].tolist()
        self.captions = dataframe['captions'].tolist()
        self.text = []

        self.txt2img = {}
        self.img2txt = {}
        self.txt2id = {}

        txt_id = 0
        for img_id in range(len(self.captions)):
            self.img2txt[img_id] = []
            for caption in self.captions[img_id]:
                self.txt2img[txt_id] = img_id
                self.img2txt[img_id].append(txt_id)
                self.text.append(caption)
                self.txt2id[txt_id] = self.identities[img_id]
                txt_id += 1
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_file_path, self.imgs[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image, idx

    def get_captions(self, img_ids):
        caption_list = []
        for img_id in img_ids:
            caption_list.append(self.captions[img_id])
        return caption_list    

    def _print_dataset_stats(self, df):
        train = df[df['split'] == 'train']
        val = df[df['split'] == 'val']
        test = df[df['split'] == 'test']

        print("--------------------------------------")
        print("subset         | # ids     | # images")
        print("--------------------------------------")
        print(f"train          |   {len(train['id'].unique())}    |    {len(train)}")
        print(f"valid          |    {len(val['id'].unique())}    |     {len(val)}")
        print(f"test           |    {len(test['id'].unique())}    |     {len(test)}")