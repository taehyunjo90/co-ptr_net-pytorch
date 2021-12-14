## Outline

It is implementation of pointer network papar. (https://arxiv.org/abs/1506.03134)

In addition to the Pointer Network proposed on this paper, I implement a basic sequential model called Seq2seq.  You can see the difference between the two models that metioned in the paper. 

For the convenience of implementation, the GRU model is used instead of the LSTM model. 



## Seq2Seq Model vs Pointer Network

![image-20211213190504710](C:\Users\TerryJo\AppData\Roaming\Typora\typora-user-images\image-20211213190504710.png)



#### Seq2seq

Seq2seq is  a very well known traditional and basic encoder-decoder model. There are two major drawbacks to solving the CO problem.



- **Generalization problem**

  The output dimensionality is fixed by the dimensionality of the problem and it is the same during training and inference. For example, model trained for 5 points of TSP problem only can solve the 5 points of TSP problem. It can't solve the problem with other than N points.

  It is because that the softmax layer of the decoder has to be fixed size. In code implementation, I called it as 'choice_size' which you can see on parameters of Seq2SeqGRU's constructor.

  

- **Bottleneck problem of context vector**

  It's a very well known problem of seq2seq model. Information loss may occur due to the limited size of the context vector. 



#### Pointer Network

Through the pointer network structure proposed in this paper, these two shortcomings of seq2seq model were solved. Especially, generalization could be achevied. It means model trained for 5 points of TSP problem could solve other than 5 points TSP problem like 7 points TSP problem, etc. If you compare seq2seq and pointer network models with the code uploaded, you can see that how generalization could works.



**We recommend to read the paper for more details. This GitHub repository is written with a focus on implementing the paper.**



## How to run this code

There are two problem situations in this code repo.

1. Sorting

   Sorting is a relatively easy-to-solve CO problem, and was implemented to quickly test the pointer network model. There is no need to download, train and test data, because data for train/test is generated in real time. 

   Run "run_seq2seq_gru_sorting.py" or "run_ptrnet_gru_sorting.py" 

2. TSP

   You have to download TSP train and test set file, which is introduced in the paper. You can download data from http://goo.gl/NDcOIG. You need put the downloaded data to "co_data" folder.

   Run "run_seq2seq_gru_tsp.py" or "run_ptrnet_gru_tsp.py" 

   

## Contact

If you have any questions about the implementation of the code at any time, please email to taehyun.jo.90@gmail.com. Thanks.

