## Pointer Network (for combinatorial optimization)



It is implementation of pointer network papar. (https://arxiv.org/abs/1506.03134)

In addition to the Pointer Network proposed on this paper, I implement a basic sequential model called Seq2seq.  You can see the difference between the two models that metioned in the paper. 

For the convenience of implementation, the GRU model is used instead of the LSTM model. 



![image-20211213190504710](C:\Users\TerryJo\AppData\Roaming\Typora\typora-user-images\image-20211213190504710.png)



## Seq2Seq Model

It's a very well known traditional and basic encoder-decoder model. There are two major drawbacks to solving the CO problem.

- **Generalization problem**

  The output dimensionality is fixed by the dimensionality of the problem and it is the same during training and inference. For example, model trained for 5 points of TSP problem only can solve the 5 points of TSP problem. It can't solve the problem with other than N points.

  It is because that the softmax layer of the decoder has to be fixed size. In code implementation, I called it as 'choice_size' which you can see on parameters of Seq2SeqGRU's constructor.

- **Bottleneck problem of context vector**

  It's a very well known problem of seq2seq model. Information loss may occur due to the limited size of the context vector. It is also the reason for the appearance of the attention model.



## Pointer Network

Through the pointer network structure proposed in this paper, these two shortcomings of Seq2seq were solved.

