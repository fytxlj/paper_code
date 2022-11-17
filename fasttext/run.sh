# 获取烹饪相关的数据集, 它是由facebook AI实验室提供的演示数据集
wget https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz && tar xvzf cooking.stackexchange.tar.gz
# 查看数据的前10条
head cooking.stackexchange.txt
# 12404条数据作为训练数据
$ head -n 12404 cooking.stackexchange.txt > cooking.train
# 3000条数据作为验证数据
$ tail -n 3000 cooking.stackexchange.txt > cooking.valid
