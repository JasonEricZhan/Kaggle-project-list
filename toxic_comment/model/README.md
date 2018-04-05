
## word embedding:
FastText:  
1.wiki.en.bin  
2.crawl-300d-2M.vec  
Glove:    
1.glove.840B.300D.txt  
2.glove.twitter.27B.200D.txt


## lb score:
### wiki.en.bin single gru:                               
public:0.9860,   private:0.9847  
### wiki.en.bin and char word2vec single gru:             
public:0.9859,   private:0.9846  
### glove.840B.300D and char word2vec single gru:         
public: 0.9855,   private:0.9847  
### glove.twitter.27B.200D LSTM attention and skip connected channel:                
public:0.9852,   private:0.9845  
### crawl-300d-2M.vec DPCNN:                              
public:0.9847,   private:0.9827  
### crawl-300d-2M.vec Capsule:                            
public:0.9847,   private:0.9841  
