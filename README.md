
# Project: ConvAI2 challenge

## Team: Enigma


### Team Members:

1- Elnaz Nouri, Microsoft Research, Elnaz.Nouri@microsoft.com, elnaaz@gmail.com 

2- Ehsan Hosseini-Asl, Salesforce Research, ehosseiniasl@salesforce.com, ehsan.hosseiniasl@gmail.com


#### Model Sources:
`projects/convai2/baselines/transformer`

`parlai/agents/transformer`

1- Model is based on transformer network
2- No ensembling is used
3- No data augmentation is used

#### Model Path:
`./checkpoints/convai2_transformer_[l=2,h=4,dw=256,dm=256,di=2048,dk=64,dv=64,src_tgt_share=False,tgt_prj=False,smooth=False]`

`valid ppl=38.65`


#### Train Command:
`./command_train_transformer.sh`
