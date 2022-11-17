import fasttext
# 针对多标签多分类问题, 使用'softmax'或者'hs'有时并不是最佳选择, 因为我们最终得到的应该是多个标签, 而softmax却只能最大化一个标签.
# 所以我们往往会选择为每个标签使用独立的二分类器作为输出层结构,
# 对应的损失计算方式为'ova'表示one vs all.
# 这种输出层的改变意味着我们在统一语料下同时训练多个二分类模型,
# 对于二分类模型来讲, lr不宜过大, 这里我们设置为0.2
model = fasttext.train_supervised(input="cooking.train", lr=0.2, epoch=25, wordNgrams=2, loss='ova')
# Read 0M words
# Number of words:  8952
# Number of labels: 735
# Progress: 100.0% words/sec/thread:   65044 lr:  0.000000 avg.loss:  7.713312 ETA:   0h 0m 0s

# 我们使用模型进行单条样本的预测, 来看一下它的输出结果.
# 参数k代表指定模型输出多少个标签, 默认为1, 这里设置为-1, 意味着尽可能多的输出.
# 参数threshold代表显示的标签概率阈值, 设置为0.5, 意味着显示概率大于0.5的标签
model.predict("Which baking dish is best to bake a banana bread ?", k=-1, threshold=0.5)
print(model.test("cooking.valid"))

# 我看到根据输入文本, 输出了它的三个最有可能的标签
# ((u'__label__baking', u'__label__bananas', u'__label__bread'), array([1.00000, 0.939923, 0.592677]))

#模型保存与加载
# 使用model的save_model方法保存模型到指定目录
# 你可以在指定目录下找到model_cooking.bin文件
model.save_model("./model_cooking.bin")

# 使用fasttext的load_model进行模型的重加载
model = fasttext.load_model("./model_cooking.bin")

# 重加载后的模型使用方法和之前完全相同
model.predict("Which baking dish is best to bake a banana bread ?", k=-1, threshold=0.5)
# ((u'__label__baking', u'__label__bananas', u'__label__bread'), array([1.00000, 0.939923, 0.592677]))
