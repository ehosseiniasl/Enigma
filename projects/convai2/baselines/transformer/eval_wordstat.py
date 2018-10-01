# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for f1 metric.
This transformer model was trained on convai2:self.
"""

from parlai.core.build_data import download_models
from parlai.scripts.eval_wordstat import setup_args, eval_wordstat
from projects.convai2.build_dict import build_dict, DICT_FILE


if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        model='transformer',
        task='convai2:self',
        external_dict=DICT_FILE,
        #model_file='models:convai2/transformer/convai2_self_transformer_model',
        #dict_file='models:convai2/transformer/convai2_self_transformer_model.dict',
        model_file='./checkpoints/convai2_transformer_[l=2,h=4,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False]',
        dict_file='./checkpoints/convai2_transformer_[l=2,h=4,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False].dict',
        dict_lower=True,
        batchsize=1,
        numthreads=1,
    )
    opt = parser.parse_args(print_args=False)
    if opt.get('model_file', '').find('convai2/transformer/convai2_self_transformer_model') != -1:
        opt['model_type'] = 'transformer'
        #fnames = ['convai2_self_transformer_model.tgz',
        #          'convai2_self_transformer_model.dict',
        #          'convai2_self_transformer_model.opt']
        fnames = ['convai2_transformer_[l=2,h=4,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False].tgz',
                'convai2_transformer_[l=2,h=4,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False].dict'
                'convai2_transformer_[l=2,h=4,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False].opt']
        download_models(opt, fnames, 'convai2', version='v3.0')
    build_dict()  # make sure true dictionary is built
    eval_wordstat(opt, print_parser=parser)
