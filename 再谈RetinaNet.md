## 再谈RetinaNet

路一直都在 [我爱计算机视觉](javascript:void(0);) *昨天*

点击我爱计算机视觉标星，更快获取CVML新技术

------



本文转载自知乎，经作者授权转载，请勿二次转载。

https://zhuanlan.zhihu.com/p/68786098

![img](https://mmbiz.qpic.cn/mmbiz_jpg/BJbRvwibeSTv91v8wI9TNTxxoT9aibe4IUClT7K7Bpf3OZTDNYHPgEp80o8GR0kjZ0ORsFtbBwIKYZYeTQIDicpbw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



Paper link:

http://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html

Code link:

https://github.com/fizyr/keras-retinanet

引

2017的ICCV中，Kaiming He大神风光一时无两，Mask R-CNN是best paper，此外FAIR的RetinaNet拿下best student paper。纵观RetinaNet论文本身，在网络结构部分并没有颠覆，之所以能够拿到best student paper，可见在其他方面的过人之处，今天我们就较为详细的探讨一下这篇论文。文章中说，RetinaNet是第一次，有一个one-stage的目标检测框架，实现和FPN，Mask R-CNN匹敌的AP。

## 一

以R-CNN系列为代表的two-stage目标检测方法在精度上已经表现很好，这类网络模型分两个步骤进行目标检测，首先选择出所有的候选区域，然后针对每个候选区域进行分类和回归。

但成也萧何败也萧何，在取得高精度的同时，two-stage的方法不能保证速度。于是另外一条道路one-stage被开辟出来，直接在原图上进行区域划分，暴力的进行分类和回归预测，这种更简单粗暴的办法在速度上加快了不少，但也正是因为不够精细，在最后的结果表现上一直落后于two-stage方法。

深究一下，到底是什么原因造成了这种精度的损失呢？一个主要原因是正负样本的不平衡，以YOLO为例，每个grid cell有5个预测，本来正负样本的数量就有差距，再相当于进行5倍放大后，这种数量上的差异更会被放大。

因此，本文基于交叉熵损失函数，提出了新的分类损失函数Focal loss，该损失函数通过抑制那些容易分类样本的权重，将注意力集中在那些难以区分的样本上，有效控制正负样本比例，防止失衡现象。



![img](https://mmbiz.qpic.cn/mmbiz_jpg/BJbRvwibeSTv91v8wI9TNTxxoT9aibe4IUVwrTdMqn5FuJibPYyZ7LKB9BaUGVMlwBJ4b0ibUFO7EXxyppQRMeLzLw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



为了验证Focal loss的有效性，设计了一个叫RetinaNet的网络进行评估。实验结果表明，RetinaNet能够在实现和one-stage同等的速度基础上，在精度上超越所有（2017年）two-stage的检测器，如下图是在COCO数据集上的实验结果：



![img](https://mmbiz.qpic.cn/mmbiz_jpg/BJbRvwibeSTv91v8wI9TNTxxoT9aibe4IUiaCT4ycbIicrGOZIbQ0bbPE3OWWV5HaBERiaxESTic1bkeF03gicEaMEe6w/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



## 二

在two-stage的检测器中，为了实现正负样本的比例均衡，不至于整个训练过程被负样本“淹没”，一般采取抽样的方法，将正负样本比例控制在1:3，采用OHEM，在正负样本间保持合理的比例。

但是传统的one-stage不行，因为one-stage是暴力粗糙的，他只有一个阶段，产生的候选相比two-stage要大得多，在实践中，通常需要大约100K个位置，这么多的位置，想想就头疼，更让人抓狂的是，这里面你真正需要的样本，少之又少。

那么我们进行抽样不行吗？杯水车薪，因为即使你抽样了，最后在训练过程中，你会惊奇的发现，整个过程还是被大量容易区分的负样本，也就是背景主导。因此，本篇论文提出了一个新的损失函数来对付类别不平衡。

如下图所示，focal loss是一个动态缩放的交叉熵损失，一言以蔽之，通过一个动态缩放因子，可以动态降低训练过程中易区分样本的权重，从而将loss的重心快速聚焦在那些难区分的样本上，而实验同样表明，相比较OHEM，sampling heuristics这些方法，focal loss更胜一筹，在以ResNet-101-FPN为backbone的RetinaNet中，AP达到了39.1，速度在5fps。



![img](https://mmbiz.qpic.cn/mmbiz_jpg/BJbRvwibeSTv91v8wI9TNTxxoT9aibe4IUGfBXg4e1Wkmb7xTOjgJibDicKIOx5KClNve5gaR3LsVubguw1abaIQAA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



## 三

下面着重说一下Focal loss

Focal loss的起源是二分类交叉熵CE，它的形式是这样的：



![img](https://mmbiz.qpic.cn/mmbiz_jpg/BJbRvwibeSTv91v8wI9TNTxxoT9aibe4IUQn8xNFEY0CJQpBtnVsiaP7Soz76GuXoj6EEZ3ZfoMsYSy0vic23bxC5w/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



关于交叉熵的解释，在我另一篇文章《关于faster r-cnn的一些思考》中有详细说明，如果你想了解更多，点我跳转

在（1）式中，y的取值有1和-1两种，代表前景和背景。p的取值范围是[0,1]，是模型预测的属于前景的概率，为了表示方便，定义一个Pt，如下所示：



![img](https://mmbiz.qpic.cn/mmbiz_jpg/BJbRvwibeSTv91v8wI9TNTxxoT9aibe4IU0R3a5xXeyd28N1tmzXzwqemdgfMOkibQ9jbfPUtficQO6fqopsQM6HUA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



综合（1）（2）就可以得到：



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



CE曲线是下图中的蓝色曲线，可以看到，相比较其他曲线，蓝色线条是变化最平缓的，即使在p>0.5(已经属于很好区分的样本)的情况下，它的损失相对于其他曲线仍然是高的，也许你会说，它相对于自己前面的已经下降很多了，对，是下降很多了，然后呢？看似每一个是微不足道，但是当数量巨大的易区分样本损失相加，就会主导你的训练过程。



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



- Balanced Cross Entropy

那怎么解决类不平衡呢？常见的思想是引入一个权重因子*α ，α* ∈[0,1]，当类标签是1是，权重因子是*α* ，当类标签是-1时，权重因子是1-*α* 。同样为了表示方便，用αt表示权重因子，那么此时的损失函数被改写为：



![img](https://mmbiz.qpic.cn/mmbiz_png/BJbRvwibeSTv91v8wI9TNTxxoT9aibe4IUNkj5CkZ1C6zwyX4d4yH40umOG10XtYPnXPF1ldwXOFjLgKrARB19Ww/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



- Focal Loss Definition

(3)式解决了正负样本的比例失衡问题（positive/negative examples），但是这种方法仅仅解决了正负样本之间的平衡问题，并没有区分简单还是难分样本（easy/hard examples），那么怎么去抑制容易区分的负样本的泛滥呢？不然整个训练过程都是围绕容易区分的样本进行，而被忽略的难区分的样本才是训练的重点。这时需要再引入一个调制因子，公式如下：



![img](https://mmbiz.qpic.cn/mmbiz_png/BJbRvwibeSTv91v8wI9TNTxxoT9aibe4IU71wu0eic1q61w2UsPCLH4Aibd7L7RJCUTYHF9HbazLwZTfuKibcd5YoHQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



*γ* 也是一个参数，范围在[0,5]，观察（4）式子可以发现，当Pt趋向于1时，说明该样本比较容易区分，整个调制因子是趋向于0的，也就是loss的贡献值会很小；如果某样本被错分，pt很小，那么此时调制因子是趋向1的，对loss没有大的影响（相对于基础的交叉熵），参数*γ* 能够调整权重衰减的速率。还是下面这张图，当*γ*=0的时候，FL就是原来的交叉熵CE，随着*γ*的增大，调整速率也在变化，实验表明，在*γ* =2时，效果最佳



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



上述两小节分别解决了正负样本不平衡问题和易分，难分样本不平衡问题，那么将这两个损失函数组合起来，就是最终的Focal loss:



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



看一下(5)式，它的功能可以解释为：通过*αt* 可以抑制正负样本的数量失衡，通过*γ* 可以控制简单/难区分样本数量失衡。关于focal loss，有以下结论：

1、无论是前景类还是背景类，*pt*越大，权重*(1-pt)r* 就越小。也就是说easy example可以通过权重进行抑制；

2、*at*用于调节positive和negative的比例，前景类别使用*at*时，对应的背景类别使用*1-at*；

3、*r*和*at*的最优值是相互影响的，所以在评估准确度时需要把两者组合起来调节。作者在论文中给出*r*=2、*at*=0.25时，ResNet-101+FPN作为backbone的结构有最优的性能。

## 四

介绍完focal loss，接下来再介绍一下验证focal loss的网络结构RetinaNet

首先，RetinaNet是一个由一个backbone和两个子网络组成的统一目标检测网络。backbone的主要作用是通过一系列卷积操作得到整张输入图像的feature map。两个子网分别基于输出的feature map进行目标分类和位置回归。整体网络结构如下图所示：



![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)



相比原版的FPN，RetinaNet的卷积过程用的是ResNet，上采样和侧边连接还是FPN结构。通过主干网络，产生了多尺度的特征金字塔。然后后面连接两个子网，分别进行分类和回归。总体来看，网络结构是非常简洁的，作者的重心并不是网路结构的创新，而是验证focal loss的有效性。

这里着重说一下anchor，这里anchor的设置类似于RPN中的结构，先看一下RPN是如何设置的：



![img](https://mmbiz.qpic.cn/mmbiz_jpg/BJbRvwibeSTv91v8wI9TNTxxoT9aibe4IUe4Y2UsicgCH6jnsmzzm0uJQdEHibwYFOJ4jcxnKssp8GjqzvibWrGFeAw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



anchor是在原图中的，对于feature map上的每一个点，利用滑窗，在原图上产生3种尺寸，每种尺寸3个长宽比的anchor，这样，一个点就会有9个anchor。对于每一个anchor，接入两个子网，分别进行1x1x18维卷积和1x1x36维卷积，这样，在分类子网，输出的是每一个anchor的前背景得分，一个点9个anchor，那么一个点就有18维度输出；

说回到RetinaNet中，在FPN的P3-P7中分别设置32x32-512x512尺寸不等的anchor，比例设置为{1:2, 1:1, 2:1}。每一层一共有9个anchor，不同层能覆盖的size范围为32-813。对每一个anchor，都对应一个K维的one-hot向量（K是类别数）和4维的位置回归向量。

与RPN相比，RetinaNet的anchor增加了多类别预测并且调整了相应的阈值。具体来说，如果一个anchor与某个GT的IOU>0.5，就认定为正样本，如果IOU在[0,0.4)之间，认定为背景。每个anchor至多用来检测一个GT，在K维标签中，将该GT对应的标签设置为1，其余归0。如果一个anchor的IOU在[0.4,0.5)之间，那么在训练时将会被忽略。位置回归计算的是anchor和某个对应GT的坐标偏移。接下来详细介绍一下两个子网：

- Classification Subnet:

分类子网对A个anchor，每个anchor中的K个类别，都预测一个存在概率。如下图所示，对于FPN的每一层输出，对分类子网来说，加上四层3x3x256卷积的FCN网络，最后一层的卷积稍有不同，用3x3xKA，最后一层维度变为KA表示，对于每个anchor，都是一个K维向量，表示每一类的概率，然后因为one-hot属性，选取概率得分最高的设为1，其余k-1为归0。传统的RPN在分类子网用的是1x1x18，只有一层，而在RetinaNet中，用的是更深的卷积，总共有5层，实验证明，这种卷积层的加深，对结果有帮助。



![img](https://mmbiz.qpic.cn/mmbiz_jpg/BJbRvwibeSTv91v8wI9TNTxxoT9aibe4IUFMcaSp1FQFLYJhXSRia5xnd0fibDwAnenvkUmzxho1q3fsOJwMTLOZPA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



- Box Regression Subnet

与分类子网并行，对每一层FPN输出接上一个位置回归子网，该子网本质也是FCN网络，预测的是anchor和它对应的一个GT位置的偏移量。首先也是4层256维卷积，最后一层是4A维度，即对每一个anchor，回归一个（x,y,w,h）四维向量。注意，此时的位置回归是类别无关的。分类和回归子网虽然是相似的结构，但是参数是不共享的。

## 五

看一下实验部分



![img](https://mmbiz.qpic.cn/mmbiz_jpg/BJbRvwibeSTv91v8wI9TNTxxoT9aibe4IUWPaAolY1lhq21hklqJTpf3CL8NEaTnW0j39God4vQo2KGiaK92esaQw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



Focal loss的威力还是很大的，当然我觉得FPN+ResNet也有加成哦！

------



**目标检测专业交流群**



关注最新最前沿的目标检测技术，欢迎加入52CV-目标检测专业交流群，扫码添加CV君拉你入群（如已为CV君好友，请直接私信，**不必重复添加**），

**（请务必注明:目标检测）：**

**![img](https://mmbiz.qpic.cn/mmbiz_png/BJbRvwibeSTs1Ke4WXicIqN7QibMXL527MCvicgajlnePVw1mnomoLqFqL0WLf7UUpSkVGj2E1GGe83e8ZmY0G42jw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)**

喜欢在QQ交流的童鞋可以加52CV官方QQ群：702781905。

（不会时时在线，如果没能及时通过还请见谅）



------

![img](https://mmbiz.qpic.cn/mmbiz_png/BJbRvwibeSTvVOnJBvePcP1qFUSWpyvrjpYAWNIZTZzUA7Zq4VPlReicJWcIeozxic5VhHlwNQNAFXmKQBtKf5xAQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

长按关注我爱计算机视觉







![img](https://mp.weixin.qq.com/mp/qrcode?scene=10000004&size=102&__biz=MzIwMTE1NjQxMQ==&mid=2247487425&idx=3&sn=8cec019e14151dae8809f3ba2fe6dfea&send_time=)

微信扫一扫
关注该公众号