import base64
import cPickle
import json
import matplotlib.pyplot as plt
import os.path as op
from tqdm import tqdm

import qd_delf.train.main
from qd_delf.train.config import config
import qd_delf.extract.extractor
from qd_delf.helper import matcher
from qd import tsv_io, qd_common

FEA_NAME_KEY = "filename"

def load_fea(fpath):
    with open(fpath, 'rb') as fp:
        return cPickle.load(fp)

def get_bbox_from_fea(fea):
    # fea["filename"] is a tuple (imgkey, bbox)
    return json.loads(json.loads(fea[FEA_NAME_KEY])[1][0])

def get_key_from_fea(fea):
    return json.loads(fea[FEA_NAME_KEY])[0][0]

def key_to_img_arr(dataset, split, key, bbox=None, enlarge=1.0):
    imgkey, _, encoded_img = dataset.seek_by_key(key, split)
    assert(imgkey == key)
    img_arr = qd_common.img_from_base64(encoded_img)
    img_h, img_w, _ = img_arr.shape
    if bbox:
        rect = qd_common.int_rect(bbox["rect"], enlarge_factor=enlarge, im_h=img_h, im_w=img_w)
        left, top, right, bot = rect
        img_arr = img_arr[top:bot, left:right]
    return img_arr

def save_img_bytes(img_bytes, fpath):
    with open(fpath, 'wb') as fp:
        fp.write(img_bytes)

def evaluate(index_fea_file, query_fea_file, pred_file,
        topk=(1, 5), cal_acc=False,
        visualize_dir=None, index_dataset_name=None, index_split="test", query_dataset_name=None, query_split="test", enlarge_bbox_factor=2.0):

    all_index_fea = load_fea(index_fea_file)
    all_query_fea = load_fea(query_fea_file)
    max_k = max(topk)

    if index_dataset_name and query_dataset_name:
        index_dataset = tsv_io.TSVDataset(index_dataset_name)
        query_dataset = tsv_io.TSVDataset(query_dataset_name)

    correct_counts = [0] * max_k
    def gen_rows():
        for query_idx, query_fea in enumerate(all_query_fea):
            scores = []
            for i, index_fea in enumerate(all_index_fea):
                inliers, locations_1_to_use, locations_2_to_use = matcher.get_inliers(
                        query_fea['location_np_list'],
                        query_fea['descriptor_np_list'],
                        index_fea['location_np_list'],
                        index_fea['descriptor_np_list'])
                if inliers is not None:
                    score = sum(inliers)
                else:
                    score = 0
                scores.append((i, score))

            scores = sorted(scores, key=lambda t: t[1], reverse=True)
            # use top1 matching image
            pred_labels = []
            for i, (fea_idx, score) in enumerate(scores):
                if i >= max_k:
                    break
                cur_pred = get_bbox_from_fea(all_index_fea[fea_idx])["class"]
                pred_labels.append((cur_pred, score))

            query_bbox = get_bbox_from_fea(query_fea)
            query_imgkey = get_key_from_fea(query_fea)

            if cal_acc:
                gt_label = query_bbox["class"]
                for i in range(max_k):
                    cur_pred = pred_labels[i][0]
                    if cur_pred == gt_label:
                        correct_counts[i] += 1
                        break

            if visualize_dir and query_idx<50:
                matched_fea = all_index_fea[scores[0][0]]
                matched_imgkey = get_key_from_fea(matched_fea)
                matched_bbox = get_bbox_from_fea(matched_fea)
                matched_img_arr = key_to_img_arr(index_dataset, index_split, matched_imgkey,
                        bbox=matched_bbox, enlarge=enlarge_bbox_factor)
                query_img_arr = key_to_img_arr(query_dataset, query_split, query_imgkey,
                        bbox=query_bbox, enlarge=enlarge_bbox_factor)
                try:
                    side_by_side_comp_img_byte, score = matcher.get_ransac_image_byte(
                            query_img_arr,
                            query_fea['location_np_list'],
                            query_fea['descriptor_np_list'],
                            matched_img_arr,
                            matched_fea['location_np_list'],
                            matched_fea['descriptor_np_list'])
                except:
                    continue
                query_att = matcher.get_attention_image_byte(query_fea['attention_np_list'])
                matched_att = matcher.get_attention_image_byte(matched_fea['attention_np_list'])
                save_img_bytes(side_by_side_comp_img_byte, op.join(visualize_dir, "{}_matchscore{}.jpg".format(query_idx, score)))
                save_img_bytes(query_att, op.join(visualize_dir, "{}att_{}.jpg".format(query_idx, query_imgkey)))
                save_img_bytes(matched_att, op.join(visualize_dir, "{}att_{}.jpg".format(query_idx, matched_imgkey)))

            yield query_imgkey, json.dumps(query_bbox), ';'.join([':'.join([it1, str(it2)]) for it1, it2 in pred_labels])

    tsv_io.tsv_writer(gen_rows(), pred_file)
    for i in range(1, len(correct_counts)):
        correct_counts[i] += correct_counts[i-1]
    return [c / float(len(all_query_fea)) for c in correct_counts]

def parse_matching_res(match_file, pred_file, topk_acc=None):
    if topk_acc:
        correct_counts = [0] * max(topk_acc)
    for parts in tsv_io.tsv_reader(match_file):
        key = parts[0]
        query_bbox = json.loads(parts[1])
        pred_labels = parts[2].split(';')


if __name__ == "__main__":
    # qd_delf.train.main.main(config)

    # model_name = config.expr
    model_name = "delf_brandsports_bneval"
    query_data_cfg = "data/brand_output/configs/logo40.yaml"
    query_fea_file = 'output/{}/delf.batch/logo40_query.delf'.format(model_name)
    index_data_cfg = "data/brand_output/configs/logo40can2.yaml"
    index_fea_file = 'output/{}/delf.batch/logo40_index.delf'.format(model_name)

    # import shutil
    # shutil.copyfile('output/{}/keypoint/ckpt/bestshot.pth.tar'.format(model_name), 'output/{}/keypoint/ckpt/fix.pth.tar'.format(model_name))
    # qd_delf.extract.extractor.main("pca", model_name, config.arch, config.target_layer, "", "")

    query_pred_file = 'output/{}/logo40_delf_query.tsv'.format(model_name)
    # qd_delf.extract.extractor.main("delf", model_name, config.arch, config.target_layer, query_data_cfg, query_fea_file)
    # qd_delf.extract.extractor.main("delf", model_name, config.arch, config.target_layer, index_data_cfg, index_fea_file)

    visualize_dir = "output/{}/visual/".format(model_name)
    qd_common.ensure_directory(visualize_dir)
    accuracy = evaluate(index_fea_file, query_fea_file, query_pred_file, cal_acc=True,
        visualize_dir=visualize_dir, index_dataset_name="logo40can2", index_split="train",
        query_dataset_name="logo40", query_split="test", enlarge_bbox_factor=2.0)
    print(accuracy)
    with open('output/{}/logo40_delf_acc.tsv'.format(model_name), 'a') as fp:
        fp.write('\t'.join([str(i) for i in accuracy]))