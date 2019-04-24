import qd_delf.train.main
from qd_delf.train.config import config
import qd_delf.extract.extractor

# qd_delf.train.main.main(config)

model_name = 'brandsports_bneval'
query_data_cfg = "data/brand_output/configs/logo40.yaml"
query_fea_file = 'repo/{}/delf.batch/logo40_query.delf'.format(model_name)
index_data_cfg = "data/brand_output/configs/logo40can2.yaml"
index_fea_file = 'repo/{}/delf.batch/logo40_index.delf'.format(model_name)

query_pred_file = 'repo/{}/logo40_delf_query.tsv'.format(model_name)

# qd_delf.extract.extractor.main("delf", model_name, query_data_cfg, query_fea_file)
# qd_delf.extract.extractor.main("delf", model_name, index_data_cfg, index_fea_file)

def test(pred_file, topk=(1, 5)):
    import cPickle
    import json
    from qd_delf.helper import matcher
    from qd import tsv_io

    def load_fea(fpath):
        with open(fpath, 'rb') as fp:
            return cPickle.load(fp)
    def get_bbox_from_fea(fea):
        return json.loads(json.loads(fea["filename"])[1][0])
    def get_key_from_fea(fea):
        return json.loads(fea["filename"])[0][0]

    all_index_fea = load_fea(index_fea_file)
    all_query_fea = load_fea(query_fea_file)
    max_k = max(topk)

    correct_counts = [0] * max_k
    def gen_rows():
        for query_fea in all_query_fea:
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
            gt_label = query_bbox["class"]

            for i in range(max_k):
                cur_pred = pred_labels[i][0]
                if cur_pred == gt_label:
                    correct_counts[i] += 1
            yield query_imgkey, json.dumps(query_bbox), ';'.join([':'.join([it1, str(it2)]) for it1, it2 in pred_labels])

    tsv_io.tsv_writer(gen_rows(), pred_file)
    for i in range(1, len(correct_counts)):
        correct_counts[i] += correct_counts[i-1]
    print([c / float(len(all_query_fea)) for c in correct_counts])
    # import ipdb; ipdb.set_trace()

test(query_pred_file)