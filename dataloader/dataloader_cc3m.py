from datasets import load_dataset

import os
from torch.utils.data import Dataset
from PIL import Image


class CC3m_Dataset(Dataset):
    def __init__(self, args, preprocess, tokenizer, subset='train', logger=None):
        logger.info("========== Initial the %s set ==========", subset)
        self.args = args

        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.num_anns = 1
        self.images_id, self.captions = [], []
        if subset != "train":
            subset = "validation"
            path = '/home/u5649209/workspace/OSA/data/cc3m-validation-0000.tar'
            self.dataset = load_dataset("pixparse/cc3m-wds", data_files={"validation": path}, split="validation")
        else:
            path = '/home/u5649209/workspace/OSA/data/cc3m-train-0000.tar'
            self.dataset = load_dataset("pixparse/cc3m-wds", data_files={"train": path}, split="train")
        # self.iter_dataset = load_dataset("pixparse/cc3m-wds", split=subset, streaming=True)
        for data_pair in self.dataset:
            self.captions.append(data_pair["txt"])
            self.images_id.append(int(data_pair["__key__"]))
            # self.image_name.append(data_pair["__key__"] + '.jpg')
        self.texts  = self.tokenizer(self.captions, truncate=True)
        logger.info('cc3m is loaded')

    def __len__(self):
        if self.subset == "train":
            return self.dataset.num_rows

    def __getitem__(self, idx):
        # image = self.preprocess(Image.open(os.path.join(self.image_root, self.image_name[idx]))) # Image from PIL module
        # text = self.texts[idx]
        # img_id = self.images_id[idx]
        image = self.preprocess(self.dataset[idx]["jpg"])
        text = self.texts[idx]
        img_id = self.images_id[idx]

        return image, text, img_id
