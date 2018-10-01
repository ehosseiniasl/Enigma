# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""Train model for ppl metric with pre-selected parameters.
These parameters have some variance in their final perplexity, but they were
used to achieve the pre-trained model.
"""

from parlai.scripts.train_model import setup_args, TrainLoop


if __name__ == '__main__':
    parser = setup_args()
    parser.set_defaults(
        task='convai2:self',
        model='transformer',
        model_file='./checkpoints/convai2_transformer_[l=2,h=4,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False]',
        dict_lower=True,
        dict_include_valid=False,
        dict_maxexs=-1,
        datatype='train',
        batchsize=64,
        hiddensize=1024,
        embeddingsize=256,
        attention='general',
        numlayers=2,
        rnn_class='lstm',
        learningrate=3,
        dropout=0.1,
        gradient_clip=0.1,
        lookuptable='enc_dec',
        optimizer='sgd',
        embedding_type='glove',
        momentum=0.9,
        bidirectional=False,
        context_length=-1,
        validation_every_n_secs=90,
        validation_metric='ppl',
        validation_metric_mode='min',
        validation_patience=12,
        log_every_n_secs=10,
        dict_tokenizer='split',
        tensorboard_log=True,
        save_every_n_sec=1800,
        max_token_seq_len=1000,
        d_word_vec=256,
        d_model = 256,
        d_inner = 2048,
        n_head = 4,
        d_k = 64,
        d_v = 64,
        )
    TrainLoop(parser).train()
