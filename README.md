由BERT到Bi-LSTM的知识蒸馏
============


整体原理介绍
------------

本例是将特定任务下BERT模型的知识蒸馏到基于Bi-LSTM的小模型中，主要参考论文 `Distilling Task-Specific Knowledge from BERT into Simple Neural Networks`实现。整体原理如下：
    
1. 在本例中，较大的模型是BERT被称为教师模型，Bi-LSTM被称为学生模型。

2. 小模型学习大模型的知识，需要小模型学习蒸馏相关的损失函数。在本实验中，损失函数是均方误差损失函数，传入函数的两个参数分别是学生模型的输出和教师模型的输出。

3. 在论文的模型蒸馏阶段，作者为了能让教师模型表达出更多的“暗知识”(dark knowledge，通常指分类任务中低概率类别与高概率类别的关系)供学生模型学习，对训练数据进行了数据增强。通过数据增强，可以产生更多无标签的训练数据，在训练过程中，学生模型可借助教师模型的“暗知识”，在更大的数据集上进行训练，产生更好的蒸馏效果。本文的作者使用了三种数据增强方式，分别是：

   - Masking: 以p-mask的概率，随机地将一个词替换为``[MASK]``，在student模型里就是``[UNK]``，而在bert中就是``[MASK]``。这个规则能够clarify每个词对label的贡献，例如，teacher网络对于``"I [MASK] the comedy"``产生的logits比``"I loved the comedy"``产出的logits要低。
   - POS-guided word replacement: 以p-pos的概率，随机地把一个词替换成相同POS(part-of-speech) tag的另一个词（如，把how替换成what）。为了保持原始的训练集的分布，新词从使用POS tag进行re-normalize的unigram的分布中采样出来。
   - N-gram sampling: 以p-ng的概率，从{1,2,…,5}中随机选一个n，然后随机采样出一个ngram。这种方法相当于随机扔掉句子的其他部分，是一种更aggressive的masking。
