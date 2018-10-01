
GPU=5
CHECKPOINT=/export/home/ehsan/ParlAI/checkpoints
LAYER=4
HEAD=2
DK=64
DV=64
D_MODEL=256
D_INNER=2048
D_WORD=256
SRC_TGT=False
TGT_PRJ=False
SMOOTH=False
MAX_LEN=600
EPOCHS=20000

EXPERIMENT=convai2_transformer_volta_[l=${LAYER},h=${HEAD},dw=${D_WORD},dm=${D_MODEL},di=${D_INNER},dk=${DK},dv=${DV},src_tgt_share=${SRC_TGT},tgt_prj=${TGT_PRJ},smooth=${SMOOTH}]
MODEL_FILE=$CHECKPOINT/$EXPERIMENT
TAG=model,numlayers,n_head,d_word_vec,d_model,d_inner,d_k,d_v

CUDA_VISIBLE_DEVICES=$GPU python projects/convai2/baselines/transformer/train.py --numworkers 4 -eps $EPOCHS --model_file $MODEL_FILE --tensorboard_tag $TAG --numlayers $LAYER --n_head $HEAD --d_word_vec $D_WORD --d_model $D_MODEL --d_inner $D_INNER --d_k $DK --d_v $DV --max_token_seq_len $MAX_LEN --src_tgt_weight_share $SRC_TGT --tgt_prj_weight_share $TGT_PRJ --label_smoothing $SMOOTH --embedding_type glove 
