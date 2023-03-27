# Unified Instance and Knowledge Alignment Pretraining for Aspect-based Sentiment Analysis

This is the official implementation of our paper, "[Unified Instance and Knowledge Alignment Pretraining for Aspect-based Sentiment Analysis](https://arxiv.org/pdf/2110.13398.pdf)" (in Pytorch).
___

## Requirements

* python 3.7
* pytorch-gpu 1.7 
* numpy 1.19.4
* pytorch_pretrained_bert 0.6.2
* nltk 3.3 
* GloVe.840B.300d
* bert-base-uncased

## Environment

- OS: Ubuntu-16.04.1
- GPU: GeForce RTX 2080
- CUDA: 10.2
- cuDNN: v8.0.2

## Dataset

1. #### target datasets

    * raw data: "./dataset/"
    * processing data: "./dataset_npy/"
    * word embedding file: "./embeddings/"

2. #### pretraining datasets
   
    * Amazon review: [Amazon Reviews for Sentiment Analysis | Kaggle](https://www.kaggle.com/bittlingmayer/amazonreviews)
    * Yelp review: [Yelp Review Sentiment Dataset | Kaggle](https://www.kaggle.com/ilhamfp31/yelp-review-dataset)
    * For the first time, please run "python ./process_data.py" to process the pretraining datasets (remember modifying the path).

## Training options

- **ds_name**: the name of target dataset, ['14semeval_laptop', '14semeval_rest', 'Twitter'], default='14semeval_rest'
- **pre_name**: the name of pretraining dataset, ['Amazon', 'Yelp'], default='Amazon'
- **bs**: batch size to use during training, [64, 100, 200], default=64
- **learning_rate**: learning rate to use, [0.001, 0.0005, 0.00001], default=0.001
- **n_epoch**: number of epoch to use, [5, 10], default=10
- **model**: the name of model, ['ABGCN', 'GCAE', 'ATAE'], default='ABGCN'
- **is_test**:  train or test the model, [0, 1], default=1
- **is_bert**: GloVe-based or BERT-based, [0, 1], default=0
- **alpha**: value of parameter \alpha in knowledge guidance loss of the paper, [0.5, 0.6, 0.7], default=0.06
- **stage**: the number of training stage, [1, 2, 3, 4], default=4

## Running

1. #### running for the first stage (pretraining on the document) 
   
    * python ./main.py -pre_name Amaozn -bs 256 -learning_rate 0.0005 -n_epoch 10 -model ABGCN -is_test 0 -is_bert 0 -stage 1 


2. #### running for the second stage
   
    * python ./main.py -ds_name 14semeval_laptop -bs 64 -learning_rate 0.001 -n_epoch 5 -model ABGCN -is_test 0 -is_bert 0 -alpha 0.6 -stage 2  
    
3. #### runing for the final stage 
   
    * python ./main.py -ds_name 14semeval_laptop -bs 64 -learning_rate 0.001 -n_epoch 10 -model ABGCN -is_test 0 -is_bert 0 -stage 3
    
4. #### training from scratch: 
   
    * python ./main.py -ds_name 14semeval_laptop -bs 64 -learning_rate 0.001 -n_epoch 10 -model ABGCN -is_test 0 -is_bert 0 -stage 4

## Evaluation

To have a quick look, we saved the best model weight trained on the target datasets in the "./best_model_weight". You can easily load them and test the performance. Due to the limited file space, we only provide the weight of ABGCN on 14semeval_laptop and 14semeval_rest datasets. You can evaluate the model weight with:

- python ./main.py -ds_name 14semeval_laptop -bs 64  -model ABGCN -is_test 1 -is_bert 0 
- python ./main.py -ds_name 14semeval_rest-bs 64  -model ABGCN -is_test 1 -is_bert 0 

## Notes

- The target datasets and more than 50% of the code are borrowed from TNet-ATT (Tang et.al, ACL2019).

- The pretraining datasets are obtained from www.Kaggle.com.

## Citation
If you find this work helpful, please consider citing as follows:  

```ruby
@article{liu2021unified,
  title={Unified instance and knowledge alignment pretraining for aspect-based sentiment analysis},
  author={Liu, Juhua and Zhong, Qihuang and Ding, Liang and Jin, Hua and Du, Bo and Tao, Dacheng},
  journal={arXiv preprint arXiv:2110.13398},
  year={2021}
}
```
