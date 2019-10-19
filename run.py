import base64
try:
    import cPickle as pickle
except ImportError:
    import pickle
import json
import matplotlib.pyplot as plt
import os.path as op
from tqdm import tqdm

import qd_delf.train.main
from qd_delf.train.config import config
import qd_delf.extract.extractor
from qd_delf.helper import matcher

from qd import tsv_io, qd_common
from qd.deteval import deteval

FEA_NAME_KEY = "filename"

class DelfFeatureFile(tsv_io.TSVFile):
    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[idx]
        except:
            raise
        self._fp.seek(pos)
        ret = pickle.loads(base64.b64decode(self._fp.readline()))
        return ret


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

def evaluate(index_fea_file, query_fea_file, outfile,
        topk=(1, 5), visualize_dir=None,
        index_dataset_name=None, index_split="test", query_dataset_name=None, query_split="test", enlarge_bbox_factor=2.0):

    # all_index_fea = DelfFeatureFile(index_fea_file)
    index_fea_tsv = DelfFeatureFile(index_fea_file)
    all_index_fea = {i: index_fea_tsv[i] for i in range(len(index_fea_tsv))}
    all_query_fea = DelfFeatureFile(query_fea_file)
    max_k = max(topk)

    # if op.isfile(outfile):
    #     raise ValueError("already exists: {}".format(outfile))
    from pathos.multiprocessing import ProcessPool as Pool
    num_worker = 16
    num_tasks = num_worker * 3
    num_rows = len(all_query_fea)
    num_rows_per_task = (num_rows + num_tasks - 1) // num_tasks
    all_args = []
    tmp_outs = []
    for i in range(num_tasks):
        cur_idx_start = i*num_rows_per_task
        if cur_idx_start >= num_rows:
            break
        cur_idx_end = min(cur_idx_start+num_rows_per_task, num_rows)
        if cur_idx_end > cur_idx_start:
            cur_outfile = outfile + "{}.{}".format(i, num_tasks)
            tmp_outs.append(cur_outfile)
            all_args.append((range(cur_idx_start, cur_idx_end), all_query_fea, all_index_fea, cur_outfile, max_k))

    # NOTE: feature matching takes a lot of time
    # _delf_feature_match(all_args[0])
    # m = Pool(num_worker)
    # m.map(_delf_feature_match, all_args)
    # m.close()
    # qd_common.concat_files([f for f in tmp_outs if op.isfile(f)], outfile)
    # for fpath in tmp_outs:
    #     tsv_io.rm_tsv(fpath)
    qd_common.concat_files([f+".tmp" for f in tmp_outs if op.isfile(f+".tmp")], outfile)

    # if visualize_dir:
    #     index_dataset = tsv_io.TSVDataset(index_dataset_name)
    #     query_dataset = tsv_io.TSVDataset(query_dataset_name)
    #     for i, parts in enumerate(tsv_io.tsv_reader(outfile)):
    #         if i >= 50:
    #             break
    #         query_fea_idx = int(parts[0])
    #         pred_labels = json.loads(parts[2])
    #         matched_fea_idx = pred_labels[0][2]
    #         visualize_feature_matching(query_fea_idx, matched_fea_idx, all_query_fea, all_index_fea, visualize_dir,
    #                 query_dataset, index_dataset, enlarge_bbox_factor)

    return calculate_accuracy_for_matching(outfile, topk)


def _delf_feature_match(args):
    query_fea_rows, all_query_fea, all_index_fea, outfile, max_k = args

    # resume from last checkpoint
    last_cache = {}
    checkpoints = [outfile + ".tmp", outfile]
    for cache_file in checkpoints:
        if op.isfile(cache_file):
            for parts in tsv_io.tsv_reader(cache_file):
                if len(parts) == 3:
                    try:
                        json.loads(parts[1])
                        json.loads(parts[2])
                    except Exception:
                        continue
                    last_cache[int(parts[0])] = parts

    def gen_rows():
        for query_idx in query_fea_rows:
            print(query_idx)
            if query_idx in last_cache:
                yield last_cache[query_idx]
            else:
                query_fea = all_query_fea[query_idx]
                scores = []
                for i in range(len(all_index_fea)):
                    index_fea = all_index_fea[i]
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
                for i, (matched_fea_idx, score) in enumerate(scores):
                    if i >= max_k:
                        break
                    cur_pred = get_bbox_from_fea(all_index_fea[matched_fea_idx])["class"]
                    pred_labels.append([cur_pred, score, matched_fea_idx])

                query_bbox = get_bbox_from_fea(query_fea)
                yield str(query_idx), qd_common.json_dump(query_bbox), qd_common.json_dump(pred_labels)

    tsv_io.tsv_writer(gen_rows(), outfile)

def calculate_accuracy_for_matching(match_res_file, topk_acc):
    max_k = max(topk_acc)
    correct_counts = [0] * max_k
    num_total = 0
    all_gt = []
    all_pred = []
    for parts in tsv_io.tsv_reader(match_res_file):
        num_total += 1
        query_fea_idx = parts[0]
        query_bbox = json.loads(parts[1])
        pred_labels = json.loads(parts[2])
        # calculate mAP
        all_gt.append([query_fea_idx, qd_common.json_dump([query_bbox])])
        all_pred.append([query_fea_idx, qd_common.json_dump(
                [{"class": pred_labels[0][0], "conf": pred_labels[0][1]/1000.0, "rect": query_bbox["rect"]}])])
        # calculate top k accuracy
        gt_label = query_bbox["class"]
        for i in range(min(max_k, len(pred_labels))):
            cur_pred = pred_labels[i][0]
            if cur_pred == gt_label:
                correct_counts[i] += 1
                break

    map_report = match_res_file + ".eval.map"
    pred_file = match_res_file + ".pred"
    gt_file = match_res_file + ".gt"
    tsv_io.tsv_writer(all_pred, pred_file)
    tsv_io.tsv_writer(all_gt, gt_file)
    deteval(truth=gt_file, dets=pred_file, report_file=map_report)

    for i in range(1, len(correct_counts)):
        correct_counts[i] += correct_counts[i-1]

    return [c / float(num_total) for c in correct_counts]

def visualize_feature_matching(query_fea_idx, index_fea_idx, all_query_fea, all_index_fea, outdir,
            query_dataset, index_dataset, enlarge_bbox_factor):
    query_fea = all_query_fea[query_fea_idx]
    index_fea = all_index_fea[index_fea_idx]
    matched_imgkey = get_key_from_fea(index_fea)
    matched_bbox = get_bbox_from_fea(index_fea)
    matched_img_arr = key_to_img_arr(index_dataset, index_split, matched_imgkey,
            bbox=matched_bbox, enlarge=enlarge_bbox_factor)
    query_imgkey = get_key_from_fea(query_fea)
    query_bbox = get_bbox_from_fea(query_fea)
    query_img_arr = key_to_img_arr(query_dataset, query_split, query_imgkey,
            bbox=query_bbox, enlarge=enlarge_bbox_factor)
    try:
        side_by_side_comp_img_byte, score = matcher.get_ransac_image_byte(
                query_img_arr,
                query_fea['location_np_list'],
                query_fea['descriptor_np_list'],
                matched_img_arr,
                index_fea['location_np_list'],
                index_fea['descriptor_np_list'])
    except Exception as e:
        print("fail to visualize: " + str(e))
        pass
    query_att = matcher.get_attention_image_byte(query_fea['attention_np_list'])
    matched_att = matcher.get_attention_image_byte(index_fea['attention_np_list'])
    save_img_bytes(side_by_side_comp_img_byte, op.join(outdir, "{}_matchscore{}.jpg".format(query_fea_idx, score)))
    save_img_bytes(query_att, op.join(outdir, "{}att_{}.jpg".format(query_fea_idx, query_imgkey)))
    save_img_bytes(matched_att, op.join(outdir, "{}att_{}.jpg".format(query_fea_idx, matched_imgkey)))


if __name__ == "__main__":
    # qd_delf.train.main.main(config)

    # model_name = config.expr
    model_name = "delf_brandsports_bneval"
    query_dataset_name = "logo200"
    query_split = "test"
    index_dataset_name = "logo200"
    index_split = "train"
    enlarge_bbox_factor = 2.0
    # query_data_cfg = "data/brand_output/configs/logo40.yaml"
    query_data_cfg = "aux_data/{}_real.yaml".format(query_dataset_name)
    query_fea_file = 'output/{}/delf.batch/{}.{}.{}.query.delf'.format(model_name, query_dataset_name, query_split, enlarge_bbox_factor)
    # index_data_cfg = "data/brand_output/configs/logo40can2.yaml"
    index_data_cfg = "aux_data/{}_cano.yaml".format(index_dataset_name)
    index_fea_file = 'output/{}/delf.batch/{}.{}.{}.index.delf'.format(model_name, index_dataset_name, index_split, enlarge_bbox_factor)

    # train PCA
    # model_path = 'output/{}/keypoint/ckpt/fix.pth.tar'.format(model_name)
    # if not op.isfile(model_path):
    #     import shutil
    #     shutil.copyfile('output/{}/keypoint/ckpt/bestshot.pth.tar'.format(model_name), model_path)
    # qd_delf.extract.extractor.main("pca", model_name, config.arch, config.target_layer, "", "")

    # extract DeLF features
    # if op.isfile(query_fea_file):
    #     raise Exception("already exist: {}".format(query_fea_file))
    # if op.isfile(index_fea_file):
    #     raise Exception("already exist: {}".format(index_fea_file))
    # qd_delf.extract.extractor.main("delf", model_name, config.arch, config.target_layer,
    #         query_data_cfg, query_fea_file, enlarge_bbox_factor)
    # qd_delf.extract.extractor.main("delf", model_name, config.arch, config.target_layer,
    #         index_data_cfg, index_fea_file, enlarge_bbox_factor)

    query_pred_file = 'output/{}/delf_query_{}_to_{}.tsv'.format(model_name, op.basename(query_fea_file), op.basename(index_fea_file))
    visualize_dir = "output/{}/visual_{}/".format(model_name, op.basename(query_pred_file))
    qd_common.ensure_directory(visualize_dir)
    accuracy = evaluate(index_fea_file, query_fea_file, query_pred_file,
        visualize_dir=visualize_dir, index_dataset_name=index_dataset_name, index_split=index_split,
        query_dataset_name=query_dataset_name, query_split=query_split, enlarge_bbox_factor=enlarge_bbox_factor)
    print(accuracy)
    with open('output/{}/delf_acc.tsv'.format(model_name), 'a') as fp:
        fp.write('\t'.join(
            [query_pred_file] \
            + [str(i) for i in accuracy]))
        fp.write('\n')
