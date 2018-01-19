# ASR+Translation

*Team : ntu_r06942112_final
*Team members :
```
r06942098 曾柏偉 	r06942112 張嘉麟
r06943124 王鈺凱	r06943121 蕭芳宗
```

## Used package
*Sklearn : from sklearn.model_selection import train_test_split SampleSubmission.csv

*jieba :  import jieba \n
	     jieba.set_dictionary(‘dict.txt.big’)

*gensim_Word2Vec : from gensim.models import Word2Vec



## Compile 
```
bash final.sh train.data train.caption test.data test.csv
```



## Model:
```
Retrieval_based + co-attention
```



## diagram
```
 (1) pad zero to the post of training/testing data
 (2) siamese network : Mfcc -> no embedding layers
					Caption -> pad_sequences -> embedding_layers
 (3) attention mechanism : dot(Mfcc, word vector of caption)
 (4) cosine similarity : dot(Mfcc,attention mechanism[flatten])
 (5) hinge loss margin : 0.2
 (6) simultaneously train contrastive and prediction model 
```
## keep in mind in the future

 *regularizer term : dropout / l2 / l2

 *output is a score : dummy variable , two models(prediction/contrastive) train simultaneously

 *Word2Vec in Chinese may not get the good word vector.when training our model, embedding layer’s trainable -> True


