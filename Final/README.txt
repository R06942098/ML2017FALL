# ASR+Translation

Team : ntu_r06942112_final
Team members :

```
r06942098 ´¿¬f°¶ 	r06942112 ±i¹ÅÅï
r06943124 ¤ýà±³Í	r06943121 ¿½ªÚ©v
```

## Used package
Sklearn : from sklearn.model_selection import train_test_split SampleSubmission.csv
jieba : import jieba
	  jieba.set_dictionary(¡¥dict.txt.big¡¦)

gensim_Word2Vec : from gensim.models import Word2Vec

##compile 
```
bash final.sh train.data train.caption test.data test.csv
```



## Model:
```
Retrieval_based + attention
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

 (1) regularizer term : dropout / l2 / l2
 (2) output is a score : dummy variable , two models(prediction/contrastive) train simultaneously
 (3)Word2Vec in Chinese may not get the good word vector.when training our model, embedding layer¡¦s trainable -> True
