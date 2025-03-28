import os
from torch.utils.data import DataLoader
from dataloader.dataloader_coco import MSCOCO_Dataset
from dataloader.dataloader_f30k import F30K_Dataset
from dataloader.dataloader_cc import CC_Dataset
from dataloader.dataloader_cc3m import CC3m_Dataset

def prepare_coco_dataloaders(args,
                             dataset_root,
                             preprocess,
                             tokenizer,
                             logger):
    """Prepare MS-COCO Caption train / val / test dataloaders
    Args:
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        preprocess: preprocess function for images
        tokenizer: the tokenizer used to encode captions
        logger: logger
    Returns:
        dataloaders (dict): keys = ["train", "val", "test"], values are the corresponding dataloaders.
    """

    image_root = os.path.join(dataset_root, 'images/')
    ann_root = os.path.join(dataset_root, 'annotations/')

    dataloaders = {}

    if args.eval:
        dataloaders['train'] = None, None
    else:
        dataloaders['train'] = dataloader_mscoco_train(
            args, image_root, ann_root, preprocess, tokenizer,
            None, 'train', logger,
        )

    dataloaders['val'] = dataloader_mscoco_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'dev', logger,
    )

    dataloaders['test'] = dataloader_mscoco_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'test', logger,
    )

    return dataloaders

def dataloader_mscoco_train(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = MSCOCO_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    #train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        #shuffle=(train_sampler is None),
        #sampler=train_sampler,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)

def dataloader_mscoco_test(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = MSCOCO_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.eval_batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)#, train_sampler



def prepare_f30k_dataloaders(args,
                             dataset_root,
                             preprocess,
                             tokenizer,
                             logger):
    """Prepare Flickr30K train / val / test dataloaders
    Args:
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        preprocess: preprocess function for images
        tokenizer: the tokenizer used to encode captions
        logger: logger
    Returns:
        dataloaders (dict): keys = ["train", "val", "test"], values are the corresponding dataloaders.
    """

    image_root = os.path.join(dataset_root, 'images/')
    ann_root = os.path.join(dataset_root, 'annotations/')

    dataloaders = {}

    if args.eval:
        dataloaders['train'] = None, None
    else:
        dataloaders['train'] = dataloader_f30k_train( 
            args, image_root, ann_root, preprocess, tokenizer,
            None, 'train', logger, 
        ) 

    dataloaders['val'] = dataloader_f30k_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'dev', logger,
    )

    dataloaders['test'] = dataloader_f30k_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'test', logger,
    )

    return dataloaders


def dataloader_f30k_train(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = F30K_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    #train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        #shuffle=(train_sampler is None),
        #sampler=train_sampler,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)

def dataloader_f30k_test(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = F30K_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.eval_batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)#, train_sampler


def prepare_cc_dataloaders(args,
                             dataset_root,
                             preprocess,
                             tokenizer,
                             logger):
    """Prepare CC120K Caption train / val / test dataloaders
    Args:
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        preprocess: preprocess function for images
        tokenizer: the tokenizer used to encode captions
        logger: logger
    Returns:
        dataloaders (dict): keys = ["train", "val", "test"], values are the corresponding dataloaders.
    """

    image_root = os.path.join(dataset_root, 'images/')
    ann_root = os.path.join(dataset_root, 'annotations/')

    dataloaders = {}

    if args.eval:
        dataloaders['train'] = None, None
    else:
        dataloaders['train'] = dataloader_cc_train(
            args, image_root, ann_root, preprocess, tokenizer,
            None, 'train', logger, 
        ) 

    dataloaders['val'] = dataloader_cc_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'dev', logger,
    )

    dataloaders['test'] = dataloader_cc_test(
        args, image_root, ann_root, preprocess, tokenizer,
        None, 'test', logger,
    )

    return dataloaders

def dataloader_cc_train(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = CC_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    #train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        #shuffle=(train_sampler is None),
        #sampler=train_sampler,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)

def dataloader_cc_test(args, image_root, annFile, preprocess, tokenizer, ids, subset, logger):
    msrvtt_dataset = CC_Dataset(
                                    args,
                                    image_root,
                                    annFile,
                                    preprocess,
                                    tokenizer,
                                    ids=ids,
                                    subset=subset,
                                    logger=logger,
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.eval_batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset)#, train_sampler
import webdataset as wds

def prepare_cc3m_dataloaders(args,
                             dataset_root,
                             preprocess,
                             tokenizer,
                             logger):

    
    dataloaders = {}

    trainset, sample_num = construct_cc3m_wds(args, preprocess, tokenizer, 'train', logger)

    trainLoader = wds.WebLoader(trainset, batch_size=None, num_workers=16, shuffle=False)
    trainLoader = trainLoader.unbatched().shuffle(1000).batched(args.batch_size)
    trainLoader = trainLoader.with_epoch(2900000//args.batch_size)

    dataloaders['train'] = trainLoader, 2900000
    dataloaders['val'] = dataloader_cc3m_test(
    args, preprocess, tokenizer,'val', logger,
    )
    dataloaders['test'] = dataloaders['val']
    return dataloaders
global count
count = 0
def make_sample(sample):
    """Take a decoded sample dictionary, augment it, and return an (image, label) tuple."""
    image = sample["jpg"]
    caption = sample["txt"]
    global count
    count += 1
    if count % 10000 == 0:
        print(count)
    return image, caption
def construct_cc3m_wds(args, preprocess, tokenizer, subset, logger):

    logger.info("========== Initial the %s set ==========", subset)
    bucket = "https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main"
    trainset_url = bucket + "/cc3m-train-{0000..0005}.tar"
    # valset_url = bucket + "/cc3m-validation-{0000..0002}.tar"
    if subset == 'train':
        url = trainset_url
    else:
        url = valset_url
    dataset = wds.WebDataset(url, resampled=True, shardshuffle=True, cache_dir=args.cache_dir, nodesplitter=wds.split_by_node)
    dataset = dataset.shuffle(1000).decode("pil").map(make_sample)

    # For IterableDataset objects, the batching needs to happen in the dataset.
    dataset = dataset.batched(args.batch_size)
    
    logger.info('cc3m is loaded')

    return dataset, 2900000

def dataloader_cc3m_test(args, preprocess, tokenizer, subset, logger):
    msrvtt_dataset = CC3m_Dataset(
                                    args,
                                    preprocess,
                                    tokenizer,
                                    subset=subset,
                                    logger=logger,
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.eval_batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader#, train_sampler

def prepare_dataloaders(args,
                        dataset_root,
                        preprocess,
                        tokenizer=None,
                        logger=None,):
    if args.dataset == 'coco':
        return prepare_coco_dataloaders(args, dataset_root, preprocess, tokenizer, logger)
    if args.dataset == 'f30k':
        return prepare_f30k_dataloaders(args, dataset_root, preprocess, tokenizer, logger)
    if args.dataset == 'cc':
        return prepare_cc_dataloaders(args, dataset_root, preprocess, tokenizer, logger)
    if args.dataset == 'cc3m':
        return prepare_cc3m_dataloaders(args, dataset_root, preprocess, tokenizer, logger)