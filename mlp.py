# -*- coding: utf-8 -*- 
import cPickle
import gzip
import os
import sys
import timeit

import numpy
import theano
import theano.tensor as T

class LR(object):
    """
    逻辑回归的实现是基于博文中给出的公式，需要预先设定好参数W和b。最小化方法用的批量随机梯度下降法MSGD。
    因此传入数据是一块一块（minibatch）的。
    """
    def __init__(self, input, n_in, n_out):
        """
        初始化函数！此类实例化时调用该函数       
        
        按照Python定义类的格式给出如下定义，需要传入的参数分别为：
         
        input的类型为 TensorType，类似于形参，起象征性的作用，并不包含真实的数据；
        input传入值为 minibatch样本数据，该数据是一个m*n的矩阵。m表示此minibatch块共有m个样本；n表示每一个样本的实际数据。
      在mnist实验中，n=784=28*28，因为每一张图片是28*28像素的。
        
        n_in 的类型为 int；
        n_in 传入值为 每个输入样本的单元数(应该是图片的高*宽(28*28=784)，但是在我们的实验数据中，
      已经把图片数据矩阵存储为了行向量(784*1)，因此这个地方传入的就是数据域中的data列的长度，
      即n_in=784，具体的样本数据是传入input里面)
        
        n_out的类型为 int
        n_out传入值为 输出结果的类别数，就是数据域中的标签的范围。此处就是0-9共10个数字。所以n_out=10。就是10分类。
        """
        
        # 初始化权值矩阵
        # numpy.zeros((m,n),dtype='float32') 是产生一组 m行n列的全0矩阵，每个矩阵元素存储为float32类型。
        # shared()函数是将生成的矩阵封装为shared类型，该类型可以用于GPU加速运算，没有其他用途。
        self.W = theano.shared(
            value = numpy.zeros(
                (n_in, n_out),
                dtype = 'float32'            
            ),            
            name = 'W',
            borrow = True
        )
        
        # 初始化偏置值
        # b是一个向量，长度为n_out,就是每一种分类都有一个偏置值
        self.b = theano.shared(
            value = numpy.zeros(
                (n_out,),
                dtype = 'float32'            
            ),
            name = 'b',
            borrow = True        
        )
        
        # 计算公式(1)，具体解释见博文 http://blog.csdn.net/niuwei22007/article/details/47705081
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        # 计算公式(3)
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)
        
        # 组织模型用到的参数，即把W和b组装成list，便于在类外引用。
        self.params = [self.W, self.b]
        
        # 记录模型的具体输入数据，便于在类外引用
        self.input = input
        
    
    def negative_log_likelihood(self, y):
        """
        负对数似然函数，即代价函数。 
        
        需要传入的参数为：
         
        y 的类型为 TensorType，类似于形参，起象征性的作用，并不包含真实的数据；
        y 传入值为 input对应的标签向量，如果input的样本数为m，则input的行数就是m，那么y就是一个m行的列向量。
        """
        # 计算完整的公式（4），具体解释见博文 http://blog.csdn.net/niuwei22007/article/details/47705081 
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
    
    def errors(self, y):
        """
        误差计算函数。传入的参数参考negative_log_likehood.

        其作用就是统计预测正确的样本数占本批次总样本数的比例。               
        """
        
        # 检查 传入正确标签向量y和前面做出的预测向量y_pred是否是具有相同的维度。如果不相同怎么去判断某个样本预测的对还是不对？
        # y.ndim返回y的维数
        # raise是抛出异常
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y doesn't have the same shape as self.y_pred")
        
        # 继续检查y是否是有效数据。依据就是本实验中正确标签数据的存储类型是int
        # 如果数据有效，则计算：
        # T.neq(y1, y2)是计算y1与y2对应元素是否相同，如果相同便是0，否则是1。
        # 举例：如果y1=[1,2,3,4,5,6,7,8,9,0] y2=[1,1,3,3,5,6,7,8,9,0]
        # 则，err = T.neq(y1,y2) = [0,1,0,1,0,0,0,0,0,0],其中有3个1，即3个元素不同
        # T.mean()的作用就是求均值。那么T.mean(err) = (0+1+0+1+0+0+0+0+0+0)/10 = 0.3,即误差率为30%
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        '''
        初始化函数！HiddenLayer实例化时调用该函数。该层与输入层是全连接，激活函数为tanh。
        
        参数介绍：
        rng 类型为：numpy.random.RandomState。
        rng 功能为：rng是用来产生随机数的实例化对象。本类中用于对W进行随机数初始化。而非0值初始化。
        
        input 类型为：符号变量T.dmatrix
        input 功能为：代表输入数据(在这里其实就是传入的图片数据x,其shape为[n_examples, n_in],n_examples是样本的数量)        
        
        n_in 类型为：int
        n_in 功能为：每一个输入样本数据的长度。和LR中一样，比如一张图片是28*28=784，
                    那么这里n_in=784，意思就是把图片数据转化为1维。
        
        n_out 类型为：int
        n_out 功能为：隐层单元的个数（隐层单元的个数决定了最终结果向量的长度）
        
        activation 类型为：theano.Op 或者 function
        activation 功能为：隐层的非线性激活函数
        '''
        self.input = input
        
        # 根据博文中的介绍，W应该按照均匀分布来随机初始化，其样本数据范围为：
        # [sqrt(-6./(fin+fout)),sqrt(6./(fin+fout))]
        # 根据博文中的说明，fin很显然就是n_in了，因为n_in就是样本数据的长度，即输入层的单元个数。
        # 同样，fout就是n_out，因为n_out是隐层单元的个数。
        # rng.uniform()的意思就是产生一个大小为size的矩阵，
        # 矩阵的每个元素值最小是low，最大是high，且所有元素值是随机均匀采样。
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            # 如果激活函数是sigmoid的话，每个元素的值是tanh的4倍。
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        
        # 偏置b初始化为0，因为梯度反向传播对b无效
        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)       
    
        self.W = W
        self.b = b
        
        # 计算线性输出，即无激活函数的结果，就等于最基本的公式 f(x)=Wx+b
        # 如果我们传入了自己的激活函数，那么就把该线性输出送入我们自己的激活函数，
        # 此处激活函数为非线性函数tanh，因此产生的结果是非线性的。
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            # 这个表达式其实很简单，就是其他高级语言里边的三目运算
            # condition?"True":"false" 如果条件(activation is None)成立，
            # 则self.output=lin_ouput
            # 否则，self.output=activation(lin_output)
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params = [self.W, self.b]
        
        
class MLP(object):
    '''
    多层感知机是一个前馈人工神经网络模型。它包含一个或多个隐层单元以及非线性激活函数。
    中间层通常使用tanh或sigmoid作为激活函数，顶层（输出层）通常使用softmax作为分类器。
    '''
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        
        '''
        rng, input在前边已经介绍过。
        
        n_in : int类型，输入数据的数目，此处对应的是输入的样本数据。        
        
        n_hidden : int类型，隐层单元数目

        n_out : int类型，输出层单元数目，此处对应的是输入样本的标签数据的数目。        
        '''
        # 首先定义一个隐层，用来连接输入层和隐层。
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )
        # 然后定义一个LR层，用来连接隐层和输出层
        self.logRegressionLayer = LR(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # 规则化，常用的是L1和L2。是为了防止过拟合。
        # 其计算方式很简单。具体规则化的内容在文章下方详细说一下
        # L1项的计算公式是：将W的每个元素的绝对值累加求和。此处有2个W，因此两者相加。
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )
        # L2项的计算公式是：将W的每个元素的平方累加求和。此处有2个W，因此两者相加。
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )
        
        # 和LR一样，计算负对数似然函数,计算误差。
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        
        self.input = input
    
def load_data(dataset):
    
     ''' 
     下载数据集。如果本地有，则直接加载，如果没有，则会从官网下载。文件目录为当前目录。
     '''
     #############
     # LOAD DATA #
     #############
            
     # 如果mnist数据集不存在，则下载
     data_dir, data_file = os.path.split(dataset)
     if data_dir == "" and not os.path.isfile(dataset):
         # Check if dataset is in the data directory.
         new_path = os.path.join(
             os.path.split(__file__)[0],
         dataset
         )
         if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
             dataset = new_path
            
     if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
         import urllib
         origin = (
             'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
             )
         print 'Downloading data from %s' % origin
         urllib.urlretrieve(origin, dataset)
            
     print '... loading data'
            
     # 加载数据集，按照格式分为训练数据集、验证数据集、测试数据集
     f = gzip.open(dataset, 'rb')
     train_set, valid_set, test_set = cPickle.load(f)
     f.close()
     #train_set, valid_set, test_set format: tuple(input, target)
     #input is an numpy.ndarray of 2 dimensions (a matrix)
     #witch row's correspond to an example. target is a
     #numpy.ndarray of 1 dimensions (vector)) that have the same length as
     #the number of rows in the input. It should give the target
     #target to the example with the same index in the input.
            
     def shared_dataset(data_xy, borrow=True):
         """ 
         将数据集设置为shared类型，便于使用GPU加速。
         """
         data_x, data_y = data_xy
         shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),
                   borrow=borrow)
         shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),
                   borrow=borrow)
         # When storing data on the GPU it has to be stored as floats
         # therefore we will store the labels as ``floatX`` as well
         # (``shared_y`` does exactly that). But during our computations
         # we need them as ints (we use labels as index, and if they are
         # floats it doesn't make sense) therefore instead of returning
         # ``shared_y`` we will have to cast it to int. This little hack
         # lets ous get around this issue
         return shared_x, T.cast(shared_y, 'int32')
            
     test_set_x, test_set_y = shared_dataset(test_set)
     valid_set_x, valid_set_y = shared_dataset(valid_set)
     train_set_x, train_set_y = shared_dataset(train_set)
            
     rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
         (test_set_x, test_set_y)]
     return rval
                
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
     """
     执行训练。学习速率为0.13，最大执行迭代次数为1000，数据集为‘mnist.pkl.gz’,样本块为600个/块
     """
     datasets = load_data(dataset)

     train_set_x, train_set_y = datasets[0]
     valid_set_x, valid_set_y = datasets[1]
     test_set_x , test_set_y  = datasets[2]
            
     # 计算总样本可以分成多少个数据块，便于后期循环用。
     n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
     n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
     n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
            
     ######################
     # BUILD ACTUAL MODEL #
     ######################
     print '... building the model'
            
     # index是正在使用的样本块的下标
     index = T.lscalar()  # index to a [mini]batch
            
     # 因为LR中的input是TensorType类型，因此引用时，也需要定义一个TensorType类型
     # x表示样本的具体数据
     x = T.matrix('x')
     # 同样y也应该是一个TensorType类型，是一个向量，而且数据类型还是int，因此定义一个T.ivector。
     # 其中i表示int，vector表示向量。详细可以参考Theano教程。
     # y表示样本的标签。
     y = T.ivector('y')
     
     # 实例化随机函数生成器
     rng = numpy.random.RandomState(1234)
       
     # x就是input样本，是一个矩阵，因此定义一个T.matrix
     # n_in，n_out的取值在此不再赘述，可以翻看上边的博文。
     # 在实例化时，会自动调用LR中的__init__函数
     classifier = MLP(rng=rng, input=x, n_in=28*28, n_hidden=n_hidden, n_out=10)
            
     # 代价函数，这是一个符号变量，cost并不是一个具体的数值。当传入具体的数据后，
     # 其才会有具体的数据产生。在原代价函数的基础上加入规则参数*规则项。
     cost = (
             classifier.negative_log_likelihood(y) 
             + L1_reg * classifier.L1 
             + L2_reg * classifier.L2_sqr
     )
     # 测试模型基本不需要说太多了，主要是用来计算当前样本块的误差率；
     # 测试不需要更新数据，因此没有updates，但是测试需要用到givens来代替cost计算公式中x和y的数值。
     # 测试模型采用的数据集是测试数据集test_set_x和test_set_y，
     test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
     )
            
     # 验证模型和测试模型的不同之处在于计算所用的数据不一样，验证模型用的是验证数据集。
     validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
     )
            
     # 对W求导，只需要调用函数T.grad()函数以及指定求(偏)导对象为classifier.W
     # MLP对比LR的不同的地方就是求偏导的参数多了2个，因此这个地方用循环来做。
     # W1, b1, W2, b2存在classifier.params中。
     # 通过遍历params中的参数，以此计算出cost对它们的偏导数，存于gparams中。   
     gparams = [T.grad(cost, param) for param in classifier.params]
            
     # updates相当于一个更新器，说明了哪个参数需要更新，以及更新公式
     # 下面代码指明更新需要参数W，更新公式是(原值-学习速率*梯度值)
     # 和求导类似，这个地方也是用到了循环去更新各个参数的值。
     updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
     ]
            
     # 上边所提到的TensorType都是符号变量，符号变量只有传入具体数值时才会生成新的数据。
     # theano.function也是一个特色函数。在本实验中，它会生成一个叫train_model的函数。
     # 该函数的参数传递入口是inputs，就是将需要传递的参数index赋值给inputs
     # 该函数的返回值是通过outputs指定的，也就是返回经过计算后的cost变量。
     # 更新器updates是用刚刚定义的update
     
     # givens是一个很实用的功能。它的作用是：在计算cost时会用到符号变量x和y（x并没有显示的表达出来，
     # 函数negative_log_likehood用到了p_y_given_x，而计算p_y_given_x时用到了input，input就是x）。
     # 符号变量经过计算之后始终会有一个自身值，而此处计算cost不用x和y的自身值，那就可以通过givens里边的表达式
     # 重新指定计算cost表达式中的x和y所用的值，而且不会改变x和y原来的值。
     
     ## 举个简单的例子：
     # state = shared(0)
     # inc = T.iscalar('inc')
     # accumulator = function([inc], state, updates=[(state, state+inc)])
     # state.get_value()  #结果是array(0)，因为初始值就是0
     # accumulator(1)     #会输出结果array(0)，即原来的state是0，但是继续往下看
     # state.get_value()  #结果是array(1)，根据updates得知，state=state+inc=0+1=1
     # accumulator(300)   #会输出结果array(1)，即原来的state是1，但是继续往下看
     # state.get_value()  #结果是array(301)，根据updates得知，state=state+inc=1+300=301
     ## 此时state=301，继续做实验
     # fn_of_state = state * 2 + inc
     ## foo用来代替更新表达式中的state，即不用state原来的值，而用新的foo值，但是fn_of_state表达式不变
     # foo = T.scalar(dtype=state.dtype)
     ## skip_shared函数是输入inc和foo,输出fn_of_state,通过givens修改foo代替fn_of_state表达式中的state
     # skip_shared = function([inc, foo], fn_of_state, givens=[(state, foo)]) 
     # skip_shared(1, 3)  #会输出结果array(7)，即fn_of_state=foo * 2 + inc = 3*2+1 = 7
     ## 再来看看state的原值是多少呢？
     # state.get_value()  #会输出结果array(301)，而不是foo的值3
     ## 希望通过这个小例子能说清楚givens的作用。
     ##因为每一次都需要用新的x和y去计算cost值，而不是用原来的上一次的x和y去计算，因此需要用到givens
     train_model = theano.function(
         inputs=[index],
         outputs=cost,
         updates=updates,
         givens={
             x: train_set_x[index * batch_size: (index + 1) * batch_size],
             y: train_set_y[index * batch_size: (index + 1) * batch_size]
         }
     )
     ###############
     # TRAIN MODEL #
     ###############
     print '... training the model'
     # early-stopping parameters
     patience = 10000    # look as this many examples regardless
     patience_increase = 2  # wait this much longer when a new best is found
     #当新的验证误差是原来的0.995倍时，才会更新best_validation_loss。即误差小了，但是至少要小了0.995倍。   
     improvement_threshold = 0.995
     #这样设置validation_frequency可以保证每一次epoch都会在验证集上测试。 
     validation_frequency = min(n_train_batches, patience / 2)

     best_validation_loss = numpy.inf
     best_iter = 0
     test_score = 0.
     start_time = timeit.default_timer()


     done_looping = False
     epoch = 0
     # 以下开始循环训练。while循环由epoch控制，是迭代次数。
     # for循环由n_train_batches控制，即一次epoch迭代共循环(总样本数/样本块数=n_train_batches)次。
     # for循环里面会累加训练过的batch数iter，当iter是validation_frequency倍数时则会在验证集上测试。
     # 如果验证集的损失this_validation_loss小于之前最佳的损失best_validation_loss，则更新best_validation_loss和best_iter，同时在testset上测试。
     # 如果验证集的损失this_validation_loss小于best_validation_loss*improvement_threshold时则更新patience。
     # 当达到最大步数n_epoch时，或者patience<iter时，结束训练
     while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)
                        print '\t\tnow patience is %d' % patience
                        print '\t\tnow iter     is %d' % iter
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

     end_time = timeit.default_timer()
     print(('Optimization complete. Best validation score of %f %% '
            'obtained at iteration %i, with test performance %f %%') %
           (best_validation_loss * 100., best_iter + 1, test_score * 100.))
     print >> sys.stderr, ('The code for file ' +
                           os.path.split(__file__)[1] +
                           ' ran for %.2fm' % ((end_time - start_time) / 60.))
# 入口函数         
if __name__ == '__main__':
    test_mlp()