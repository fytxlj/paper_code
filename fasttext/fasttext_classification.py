import fasttext
# 我们这里将其设置为2意味着添加2-gram特征, 这些特征帮助模型捕捉前后词汇之间的关联, 更好的提取分类规则用于模型分类, 当然这也会增加模型训时练占用的资源和时间.
# 设置train_supervised方法中的参数loss来修改损失计算方式(等效于输出层的结构), 默认是softmax层结构
# 我们这里将其设置为'hs', 代表层次softmax结构, 意味着输出层的结构(计算方式)发生了变化, 将以一种更低复杂度的方式来计算损失.(logk)


# 因此可以使用fasttext的autotuneValidationFile参数进行自动超参数调优.


model = fasttext.train_supervised(input = "cooking.valid",epoch=25,lr=1.0,wordNgrams=2,loss='hs')

# autotuneValidationFile参数需要指定验证数据集所在路径, 它将在验证集上使用随机搜
# 使用autotuneDuration参数可以控制随机搜索的时间, 默认是300s
model = fasttext.train_supervised(input='cooking.train', autotuneValidationFile='cooking.valid', autotuneDuration=600)
print(model.predict("Which baking dish is best to bake a banana bread ?"))
# 为了评估模型到底表现如何, 我们在3000条的验证集上进行测试
print(model.test("cooking.valid"))
# 元组中的每项分别代表, 验证集样本数量, 精度以及召回率
# 我们看到模型精度和召回率表现都很差, 接下来我们讲学习如何进行优化.
# (3000, 0.124, 0.0541)

