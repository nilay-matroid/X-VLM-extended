import os
import sys
print(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from typing import List

import ruamel.yaml as yaml
import numpy as np
from tqdm import tqdm
from PIL import Image
from time import time

import torch
import torch.nn.functional as F
from torchvision import transforms

from models.model_retrieval import XVLM

from models.tokenization_bert import BertTokenizer
from models.tokenization_roberta import RobertaTokenizer
from dataset.re_dataset import re_eval_dataset
from dataset.RSTPReid import RSTPReid
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import create_loader

def setup_dataset(dataset, config, split='test'):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    if dataset == 're':
        if split == 'test':
            test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        elif split == 'val':
            test_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        else:
            raise NotImplementedError
        return test_dataset
    elif dataset == 'personreid_re':
        return RSTPReid(config['image_root'], split=split, transform=test_transform)
    else:
        raise NotImplementedError

def get_config(args):
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    return config    
    

class Visualizer:
    def __init__(self, args) -> None:
        config = get_config(args)
        torch.cuda.empty_cache() 
        print("Creating model", flush=True)
        model = XVLM(config=config)
        model.load_pretrained(args.checkpoint, config, is_eval=not args.load_pretrained)
        model = model.to(args.device)
        model.eval()

        print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        if config['use_roberta']:
            tokenizer = RobertaTokenizer.from_pretrained(config['text_encoder'])
        else:
            tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])

        print("Creating retrieval dataset", flush=True)
        test_dataset = setup_dataset(args.dataset, config, args.split)

        print("Creating Dataloader...")
        test_loader = create_loader([test_dataset], [None],
                                        batch_size=[config['batch_size_test']],
                                        num_workers=[4],
                                        is_trains=[False],
                                        collate_fns=[None])[0]

        self.model = model
        self.tokenizer = tokenizer
        self.test_loader = test_loader
        self.args = args
        self.config = config
        self.person_reid = (args.dataset == 'personreid_re')

    @torch.no_grad()
    def index_images(self) -> None:
        image_feats = []
        image_embeds = []
        img_ids = []
        for image, img_id in tqdm(self.test_loader):
            image = image.to(self.args.device)
            image_feat = self.model.vision_encoder(image)
            image_embed = self.model.vision_proj(image_feat[:, 0, :])
            image_embed = F.normalize(image_embed, dim=-1).cpu()

            image_feat = image_feat.cpu()
            image = image.cpu()


            image_feats.append(image_feat)
            image_embeds.append(image_embed)
            img_ids.append(img_id)

        self.image_feats = torch.cat(image_feats, dim=0)
        self.image_embeds = torch.cat(image_embeds, dim=0)
        self.img_ids = torch.cat(img_ids, dim=0)

    @torch.no_grad()
    def search_texts(self, texts: List[str]) -> List[List[int]]:
        num_text = len(texts)
        text_bs = self.config['batch_size_test_text']  # 256
        text_feats = []
        text_embeds = []
        text_atts = []

        for i in tqdm(range(0, num_text, text_bs)):
            text = texts[i: min(num_text, i + text_bs)]
            text_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.config['max_tokens'],
                                return_tensors="pt").to(self.args.device)
            text_output = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
            text_feat = text_output.last_hidden_state
            text_embed = F.normalize(self.model.text_proj(text_feat[:, 0, :]))
            text_embeds.append(text_embed)
            text_feats.append(text_feat)
            text_atts.append(text_input.attention_mask)

        text_embeds = torch.cat(text_embeds, dim=0).cpu()
        text_feats = torch.cat(text_feats, dim=0).cpu()
        text_atts = torch.cat(text_atts, dim=0)

        sims_matrix = self.image_embeds @ text_embeds.t()
        sims_matrix = sims_matrix.t()

        score_matrix_t2i = torch.full((len(texts), self.img_ids.shape[0]), -100.0).to(self.args.device)

        for i, sims in enumerate(sims_matrix):
            topk_sim, topk_idx = sims.topk(k=self.config['k_test'], dim=0)
            encoder_output = self.image_feats[topk_idx].to(self.args.device)
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(self.args.device)
            output = self.model.text_encoder(encoder_embeds=text_feats[i].repeat(self.config['k_test'], 1, 1).to(self.args.device),
                                        attention_mask=text_atts[i].repeat(self.config['k_test'], 1),
                                        encoder_hidden_states=encoder_output.to(self.args.device),
                                        encoder_attention_mask=encoder_att,
                                        return_dict=True,
                                        mode='fusion'
                                        )
            score = self.model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
            score_matrix_t2i[i, topk_idx] = score

        score_matrix_t2i = score_matrix_t2i.cpu().numpy()
        image_id_list = []
        for text_index, score in enumerate(score_matrix_t2i):
            image_inds_index = np.argsort(score)[::-1]
            image_ids = [self.img_ids[ind] for ind in image_inds_index]
            image_id_list.append(image_ids)

        return image_id_list


    def load_image(self, image_id: int) -> Image:
        dataset = self.test_loader.dataset
        if self.person_reid:
            image_path = os.path.join(dataset.image_file_path, dataset.imgs[image_id])
        else:
            image_path = os.path.join(dataset.image_root, dataset.ann[image_id]['image'])
        return Image.open(image_path).convert('RGB')
    

    def visualize(self, texts: List[str], image_ids: List[List[int]], max_rank: int = 10, gt_image_ids: List[List[int]] = [], gt_ids: List[List[int]] = []) -> None: 
        root_path = os.path.join(self.args.output_dir)
        if not os.path.isdir(root_path):
            os.mkdir(root_path)

        if self.person_reid and len(gt_ids) > 0:
            img2txt = self.test_loader.dataset.img2txt
            txt2id = self.test_loader.dataset.txt2id

        vis_label = len(gt_image_ids) > 0

        for textid, (text, image_id) in enumerate(zip(texts, image_ids)):
            image_id = torch.tensor(image_id).tolist()
            print("Query: ", text)
            if len(gt_image_ids) > 0:
                gt_set = set(gt_image_ids[textid])    

            if vis_label:
                if self.person_reid:
                    fig, axes = plt.subplots(2, max_rank, figsize=(3 * max_rank, 36))
                else:
                    fig, axes = plt.subplots(2, max_rank, figsize=(3 * max_rank, 12))
            else:
                fig, axes = plt.subplots(1, max_rank, figsize=(3 * max_rank, 6))

            plt.clf()

            for i in range(min(max_rank, len(image_id))):
                if vis_label:
                    ax = fig.add_subplot(1, max_rank, i + 1)
                else:
                    ax = fig.add_subplot(2, max_rank, i + 1)

                image = self.load_image(image_id[i])            
                gallery_img = np.asarray(image, dtype=np.uint8)

                is_gt = (len(gt_ids) > 0 and self.person_reid and (gt_ids[textid] == txt2id[img2txt[image_id[i]][0]]))\
                     or (len(gt_image_ids) > 0 and (not self.person_reid) and (image_id[i] in gt_set))
               
                if is_gt:
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                                height=gallery_img.shape[0] - 1, edgecolor=(0, 1, 0),
                                                fill=False, linewidth=5))
                else:
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                                height=gallery_img.shape[0] - 1, edgecolor=(0, 0, 1),
                                                fill=False, linewidth=5))

                ax.imshow(gallery_img)
                ax.axis("off")

            if vis_label:
                for index, gt_image_id in enumerate(gt_image_ids[textid]):
                    ax = fig.add_subplot(2, max_rank, max_rank + 1 + index)
                    gt_image = self.load_image(gt_image_id)            
                    gt_gallery_img = np.asarray(gt_image, dtype=np.uint8)
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gt_gallery_img.shape[1] - 1,
                                                height=gt_gallery_img.shape[0] - 1, edgecolor=(0, 1, 0),
                                                fill=False, linewidth=5))
                    ax.imshow(gt_gallery_img)
                    ax.axis("off")

            plt.tight_layout()
            filepath = os.path.join(root_path, f"{text}_{self.args.dataset}_results.jpg")
            fig.savefig(filepath)

    
    def show_results(self, query) -> None:
        gt_image_ids = []
        gt_ids = []
        if type(query) == str or type(query) == int:
            query = [query]

        if type(query) == list and type(query[0]) == int:
            if self.person_reid:
                gt_ids = [self.test_loader.dataset.txt2id[id] for id in query]
                for gt_id in gt_ids:
                    gt_image_id = []
                    for image_id in range(len(self.test_loader.dataset.imgs)):
                        if self.test_loader.dataset.identities[image_id] == gt_id:
                            gt_image_id.append(image_id)
                    gt_image_ids.append(gt_image_id)
            else:
                gt_image_ids = [[self.test_loader.dataset.txt2img[id]] for id in query]
            query = [self.test_loader.dataset.text[id] for id in query]

        start_time = time()
        image_id_list = self.search_texts(query)

        query_size = 1
        if type(query) == list:
            query_size = len(query)

        print(f"Time to fetch image ids for {query_size} queries : {time() - start_time}")
        self.visualize(texts=query, image_ids=image_id_list, max_rank=10, gt_image_ids=gt_image_ids, gt_ids=gt_ids)
