# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Evaluate pre-trained model trained for ppl metric.
This transformer model was trained on convai2:self.
"""
from parlai.core.build_data import download_models
from projects.convai2.eval_ppl import setup_args, eval_ppl



if __name__ == '__main__':
    parser = setup_args()
    parser.set_params(
        model='parlai.agents.transformer.transformer:PerplexityEvaluatorAgent',
        #model_file='models:convai2/transformer/convai2_self_transformer_model',
        model_file='./checkpoints/convai2_transformer_volta_[l=4,h=2,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False]',
        #dict_file='models:convai2/transformer/convai2_self_transformer_model.dict',
        dict_file='./checkpoints/convai2_transformer_volta_[l=4,h=2,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False].dict',
        dict_lower=True,
        batchsize=1,
        numthreads=60,
        no_cuda=True,
    )
    opt = parser.parse_args()
    #if opt.get('model_file', '').find('convai2/transformer/convai2_self_transformer_model') != -1:
    if opt.get('model_file', '').find('convai2/transformer/convai2_self_transformer_model') != -1:
        opt['model_type'] = 'transformer'
        #fnames = ['convai2_self_transformer_model.tgz',
        #          'convai2_self_transformer_model.dict',
        #          'convai2_self_transformer_model.opt'],
        fnames = ['convai2_transformer_volta_[l=4,h=2,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False].tgz',
                'convai2_transformer_volta_[l=4,h=2,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False].dict'
                'convai2_transformer_volta_[l=4,h=2,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False].opt']
        download_models(opt, fnames, 'convai2', version='v3.0')
    eval_ppl(opt)
