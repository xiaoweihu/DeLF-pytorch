
'''extractor.py
extract DeLF local features
'''

import os, sys, time
# sys.path.append('../')
# sys.path.append('../train')
import argparse
import base64

import torch
import torch.nn
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import h5py
import json
try:
    import cPickle as pickle
except ImportError:
    import pickle
import copy

from ..helper import delf_helper
from ..train.delf import Delf_V1
from ..train.dataset import CropClassTSVDatasetYaml
from .pca import DelfPCA
# from folder import ImageFolder
from ..utils import mkdir_p, AverageMeter

from qd.tsv_io import tsv_writer

__DEBUG__ = False

'''helper functions.
'''
def __cuda__(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x
    # return x

def __is_cuda__():
    return torch.cuda.is_available()
    # return False

def __to_var__(x, volatile=False):
    return Variable(x, volatile=volatile)

def __to_tensor__(x):
    return x.data

# def __build_delf_config__(data):
#     parser = argparse.ArgumentParser('delf-config')
#     parser.add_argument('--stage', type=str, default='inference')
#     parser.add_argument('--expr', type=str, default='dummy')
#     parser.add_argument('--ncls', type=str, default='dummy')
#     parser.add_argument('--use_random_gamma_rescale', type=str, default=False)
#     parser.add_argument('--arch', type=str, default=data['ARCH'])
#     parser.add_argument('--load_from', type=str, default=data['LOAD_FROM'])
#     parser.add_argument('--target_layer', type=str, default=data['TARGET_LAYER'])
#     delf_config, _ = parser.parse_known_args()

#     # print config.
#     state = {k: v for k, v in delf_config._get_kwargs()}
#     print(state)
#     return delf_config


class FeatureExtractor():
    def __init__(self,
                 extractor_config):

        # environment setting.
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(extractor_config.get('GPU_ID'))

        # parameters.
        self.title = 'DeLF-Inference'
        self.mode = extractor_config.get('MODE')
        self.ncls = extractor_config.get('NCLS')
        self.iou_thres = extractor_config.get('IOU_THRES')
        self.attn_thres = extractor_config.get('ATTN_THRES')
        self.top_k = extractor_config.get('TOP_K')
        self.target_layer = extractor_config.get('TARGET_LAYER')
        self.scale_list = extractor_config.get('SCALE_LIST')
        self.use_pca = extractor_config.get('USE_PCA')
        self.input_path = extractor_config.get('INPUT_PATH')
        self.output_path = extractor_config.get('OUTPUT_PATH')

        self.enlarge_bbox = extractor_config.get('ENLARGE_BBOX')

        # load pytorch model
        print('load DeLF pytorch model...')
        # delf_config = __build_delf_config__(extractor_config)
        self.model = Delf_V1(
            ncls = 'dummy',
            load_from = extractor_config['LOAD_FROM'],
            arch = extractor_config['ARCH'],
            stage = 'inference',
            target_layer = extractor_config['TARGET_LAYER'],
            use_random_gamma_rescale = False)
        self.model.eval()
        self.model = __cuda__(self.model)

        # load pca matrix
        if self.mode.lower() in ['delf']:
            if self.use_pca:
                print('load PCA parameters...')
                h5file = h5py.File(extractor_config.get('PCA_PARAMETERS_PATH'), 'r')
                self.pca_mean = h5file['.']['pca_mean'].value
                self.pca_vars = h5file['.']['pca_vars'].value
                self.pca_matrix = h5file['.']['pca_matrix'].value
                self.pca_dims = extractor_config.get('PCA_DIMS')
                self.use_pca = extractor_config.get('USE_PCA')
            else:
                print('PCA will not be applied...')
                self.pca_mean = None
                self.pca_vars = None
                self.pca_matrix = None
                self.pca_dims = None

        # PCA.
        if self.mode.lower() in ['pca']:
            self.pca = DelfPCA(
                pca_n_components = extractor_config.get('PCA_DIMS'),
                pca_whitening = True,
                pca_parameters_path = extractor_config.get('PCA_PARAMETERS_PATH'))

        # set receptive field, stride, padding.
        if self.target_layer in ['layer3']:
            self.rf = 291.0
            self.stride = 16.0
            self.padding = 145.0
        elif self.target_layer in ['layer4']:
            self.rf = 483.0
            self.stride = 32.0
            self.padding = 241.0
        else:
            raise ValueError('Unsupported target_layer: {}'.format(self.target_layer))

    def __adjust_pixel_range__(self,
                             x,
                             range_from=[0,1],
                             range_to=[-1,1]):
        '''
        adjust pixel range from <range_from> to <range_to>.
        '''
        if not range_from == range_to:
            scale = float(range_to[1]-range_to[0])/float(range_from[1]-range_from[0])
            bias = range_to[0]-range_from[0]*scale
            x = x.mul(scale).add(bias)
            return x


    def __extract_delf_feature__(self, x, filename, mode='pca'):
        '''extract raw features from image batch.
        x: Input FloatTensor, [b x c x w x h]
        output: Output FloatTensor, [b x c x dim x dim]
        '''
        if mode == 'pca':
            use_pca = False
            pca_mean = 'dummy_pca_mean',
            pca_vars = 'dummy_pca_vars',
            pca_matrix = 'dummy_pca_matrix',
            pca_dims = 'dummy_pca_dims',
            workers = 1
        else:
            assert mode == 'delf', 'mode must be either pca or delf'
            use_pca = copy.deepcopy(self.use_pca)
            pca_mean = copy.deepcopy(self.pca_mean)
            pca_vars = copy.deepcopy(self.pca_vars)
            pca_matrix = copy.deepcopy(self.pca_matrix)
            pca_dims = copy.deepcopy(self.pca_dims)
            workers = 1
        try:
            output = delf_helper.GetDelfFeatureFromMultiScale(
                x = x,
                model = self.model,
                filename = filename,
                pca_mean = pca_mean,
                pca_vars = pca_vars,
                pca_matrix = pca_matrix,
                pca_dims = pca_dims,
                rf = self.rf,
                stride = self.stride,
                padding = self.padding,
                top_k = self.top_k,
                scale_list = self.scale_list,
                iou_thres = self.iou_thres,
                attn_thres = self.attn_thres,
                use_pca = use_pca,
                workers = workers)
            # import ipdb; ipdb.set_trace()

            if mode == 'pca':
                descriptor_np_list = output['descriptor_np_list']
                descriptor = [descriptor_np_list[i,:] for i in range(descriptor_np_list.shape[0])]
                return descriptor
            else:
                return output
        except Exception as e:
            print('\n[Error] filename:{}, error message:{}'.format(filename, e))
            return None


    def __save_delf_features_to_file__(self,
                                       data,
                                       filename):
        '''save final local features after delf-postprocessing(PCA, NMS)
        use pickle to save features.
        Args:
            data = [{
                filename:
                location_np_list:
                descriptor_np_list:
                feature_scale_np_list:
                attention_score_np_list:
                attention_np_list:
            }, ... ]
        '''
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=2)       # use protocol <= 2 for python2 compatibility.
        print('\nsaved DeLF feature at {}'.format(filename))

    def delf_features2str(self, data):
        return base64.b64encode(pickle.dumps(data, protocol=2))


    def __save_raw_features_to_file__(self,
                                      feature_maps,
                                      filename):
        '''save feature to local file.
        feature_maps: list of descriptor tensors in batch. [x1, x2, x3, x4 ...], x1 = Tensor([c x w x h])
        output_path: path to save file.

        save:
        list of descriptors converted to numpy array. [d1, d2, d3, ...]
        '''
        np_feature_maps = []
        np_feature_maps = [x.numpy() for _, x in enumerate(feature_maps)]
        np_feature_maps = np.asarray(np_feature_maps)

        # save features, labels to h5 file.
        h5file = h5py.File(filename, 'w')
        h5file.create_dataset('feature_maps', data=np_feature_maps)
        h5file.close()


    def extract(self, input_path, output_path):
        '''extract features from single image without batch process.
        '''
        assert self.mode.lower() in ['pca', 'delf']
        batch_timer = AverageMeter()
        data_timer = AverageMeter()

        # dataloader.
        # dataset = ImageFolder(
        #     root = input_path,
        #     transform = transforms.ToTensor())
        bgr_normalize = transforms.Normalize(mean=[0.406, 0.456, 0.485],
                                     std=[0.225, 0.224, 0.229])
        transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    # transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    bgr_normalize,
                ])

        if input_path.endswith('.yaml'):
            dataset = CropClassTSVDatasetYaml(input_path, session_name='test',
                                              transform=transform,
                                              enlarge_bbox=self.enlarge_bbox)
        else:
            raise NotImplementedError()

        self.dataloader = torch.utils.data.DataLoader(
            dataset = dataset,
            batch_size = 1,
            shuffle = (not __DEBUG__),
            num_workers = 0)
        feature_maps = []
        if self.mode.lower() in ['pca']:
            # bar = Bar('[{}]{}'.format(self.mode.upper(), self.title), max=len(self.dataloader))
            for batch_idx, (inputs, fileinfos) in enumerate(self.dataloader):
                filename = json.dumps(fileinfos)
                # if filename != "https://s-media-cache-ak0.pinimg.com/600x315/44/02/42/440242443ac39b6f6d5a6f41bae2bd31.jpg":
                #     continue
                # image size upper limit.
                if not (len(inputs.size()) == 4):
                    if __DEBUG__:
                        print('wrong input dimenstion! ({},{})'.format(filename, input.size()))
                    continue;
                if not (inputs.size(2)*inputs.size(3) <= 1200*1200):
                    if __DEBUG__:
                        print('passed: image size too large! ({},{})'.format(filename, inputs.size()))
                    continue;
                if not (inputs.size(2) >= 112 and inputs.size(3) >= 112):
                    if __DEBUG__:
                        print('passed: image size too small! ({},{})'.format(filename, inputs.size()))
                    continue;

                data_timer.update(time.time() - since)
                # prepare inputs
                if __is_cuda__():
                    inputs = __cuda__(inputs)
                inputs = __to_var__(inputs)

                # get delf feature only for pca calculation.
                pca_feature = self.__extract_delf_feature__(inputs.data, filename, mode='pca')
                if pca_feature is not None:
                    feature_maps.extend(pca_feature)

                batch_timer.update(time.time() - since)
                since = time.time()

                # progress
                log_msg  = ('\n[Extract][Processing:({batch}/{size})] '+ \
                            'eta: (data:{data:.3f}s),(batch:{bt:.3f}s)') \
                .format(
                    batch=batch_idx + 1,
                    size=len(self.dataloader),
                    data=data_timer.val,
                    bt=batch_timer.val)
                    # tt=bar.elapsed_td)
                print(log_msg)
                # bar.next()
                print('\nnumber of selected features so far: {}'.format(len(feature_maps)))
                if len(feature_maps) >= 1000000:        # UPPER LIMIT.
                    break;

                # free GPU cache every.
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    if __DEBUG__:
                        print('GPU Memory flushed !!!!!!!!!')

            # trian PCA.
            self.pca(feature_maps)

        else:
            # bar = Bar('[{}]{}'.format(self.mode.upper(), self.title), max=len(self.dataloader))
            assert self.mode.lower() in ['delf']
            feature_maps = []
            def gen_rows():
                since = time.time()
                for batch_idx, (inputs, fileinfos) in enumerate(self.dataloader):
                    filename = json.dumps(fileinfos)
                    # image size upper limit.
                    if not (len(inputs.size()) == 4):
                        # if __DEBUG__:
                        print('wrong input dimenstion! ({},{})'.format(filename, inputs.size()))
                        continue;
                    if not (inputs.size(2)*inputs.size(3) <= 1200*1200):
                        # if __DEBUG__:
                        print('passed: image size too large! ({},{})'.format(filename, inputs.size()))
                        continue;
                    if not (inputs.size(2) >= 10 and inputs.size(3) >= 10):
                        # if __DEBUG__:
                        print('passed: image size too small! ({},{})'.format(filename, inputs.size()))
                        continue;

                    data_timer.update(time.time() - since)
                    # prepare inputs
                    if __is_cuda__():
                        inputs = __cuda__(inputs)
                    # inputs = __to_var__(inputs)

                    # get delf everything (score, feature, etc.)
                    # delf_feature = self.__extract_delf_feature__(inputs.data, filename, mode='delf')
                    delf_feature = self.__extract_delf_feature__(inputs.detach(), filename, mode='delf')
                    if delf_feature is not None:
                        # feature_maps.append(delf_feature)
                        yield [self.delf_features2str(delf_feature)]

                    # log.
                    batch_timer.update(time.time() - since)
                    since = time.time()
                    log_msg  = ('\n[Extract][Processing:({batch}/{size})] '+ \
                                'eta: (data:{data:.3f}s),(batch:{bt:.3f}s)') \
                    .format(
                        batch=batch_idx + 1,
                        size=len(self.dataloader),
                        data=data_timer.val,
                        bt=batch_timer.val)
                        # tt=bar.elapsed_td)
                    print(log_msg)
                    # bar.next()

                    # free GPU cache every.
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        if __DEBUG__:
                            print('GPU Memory flushed !!!!!!!!!')

            # use pickle to save DeLF features.
            # self.__save_delf_features_to_file__(feature_maps, output_path)
            tsv_writer(gen_rows(), output_path)


def main(MODE, model_name, arch, target_layer, input_path, output_path, enlarge_bbox):
    # MODE = 'delf'           # either "delf" or "pca"
    # MODE = 'pca'
    # GPU_ID = 0
    IOU_THRES = 0.98
    ATTN_THRES = 0.17
    TOP_K = 1000
    USE_PCA = True
    PCA_DIMS = 40
    SCALE_LIST = [0.25, 0.3535, 0.5, 0.7071, 1.0, 1.4142, 2.0]

    # MODEL_NAME = 'ldmk'

    LOAD_FROM = 'output/{}/keypoint/ckpt/fix.pth.tar'.format(model_name)
    PCA_PARAMETERS_PATH = 'output/{}/pca.h5'.format(model_name)
    DATASET_PATH = "data/brand_output/configs/train_nobg.yamllst"

    extractor_config = {
        # params for feature extraction.
        'MODE': MODE,
        # 'GPU_ID': GPU_ID,
        'IOU_THRES': IOU_THRES,
        'ATTN_THRES': ATTN_THRES,
        'TOP_K': TOP_K,
        'PCA_PARAMETERS_PATH': PCA_PARAMETERS_PATH,
        'PCA_DIMS': PCA_DIMS,
        'USE_PCA': USE_PCA,
        'SCALE_LIST': SCALE_LIST,

        # params for model load.
        'LOAD_FROM': LOAD_FROM,
        'ARCH': arch,
        'EXPR': model_name,
        'TARGET_LAYER': target_layer,

        'ENLARGE_BBOX': enlarge_bbox,
    }


    extractor = FeatureExtractor(extractor_config)
    if MODE.lower() in ['pca']:
        extractor.extract(DATASET_PATH, PCA_PARAMETERS_PATH)

    elif MODE.lower() in ['delf']:
        # query
        if os.path.isfile(output_path):
            raise ValueError("already exists: {}".format(output_path))
        mkdir_p(os.path.dirname(output_path))
        extractor.extract(input_path, output_path)
        # index
        # input_path = "data/brand_output/configs/logo40can2.yaml"
        # output_path = 'repo/{}/delf.batch/logo40_index.delf'.format(MODEL_NAME)
        # extractor.extract(input_path, output_path)





