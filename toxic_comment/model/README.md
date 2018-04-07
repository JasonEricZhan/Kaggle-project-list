
## word embedding:
FastText:  
1.wiki.en.bin  
2.crawl-300d-2M.vec  
Glove:    
1.glove.840B.300D.txt  
2.glove.twitter.27B.200D.txt


## lb score(mean columnwise AUC):
#### wiki.en.bin single gru:                               
public:0.9860,   private:0.9847  
#### wiki.en.bin and char word2vec single gru:             
public:0.9859,   private:0.9846  
#### glove.840B.300D and char word2vec single gru:         
public: 0.9855,   private:0.9847  
#### glove.twitter.27B.200D LSTM attention and skip connected channel:                
public:0.9852,   private:0.9845  
#### crawl-300d-2M.vec DPCNN:                              
public:0.9847,   private:0.9827  
#### crawl-300d-2M.vec Capsule:                            
public:0.9847,   private:0.9841  

#### final ensemble:
stacking with 7 models(one from teamate John Miller, lgbm lb public score: 0.9820)

ensemble step:  
1.average high correlated model(mean correlate of each corresponded column bigger than 0.98) :
* 0.5 * DPCNN+ 0.5 * Capsule, and get the new out of fold (new out of fold--no.1)   
* 0.5 * Single gru+ 0.5 * Wiki.en.bin and char word2vec single gru, and get the new out of fold (new out of fold--no.2)  

2.stacking 5 out of fold:  
* new out of fold--no.1  
* new out of fold--no.2     
* LSTM attention    
* glove and char single gru    
* lgbm (from teamate John Miller)    
