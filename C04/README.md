邮件分类classify_email.py
(1) 收集数据:提供文本文件。email/ham以及email/spam文件夹中
(2) 准备数据:将文本文件解析成词条向量。
            classify_email.py中的dataparse函数进行数据解析
            doc_process.py中的create_vocabulary函数生成词汇表
            doc_process.py中的wordset2vec/wordbag2vec函数生成词向量
(3) 分析数据:检查词条确保解析的正确性。
(4) 训练算法:naive_bayes.py中的train_nbayes函数。任意数量的类别都适用。
(5) 测试算法:naive_bayes.py中的classify_nb函数,并且构建一个新的测试函数来计算文档集的错误率。
(6) 使用算法:classify_email.py，　并查找了每个类别的高频词汇

存在的问题：
1.classify_email.py中将解析文件时，三个文件“./email/ham/6.txt”、“./email/spam/17.txt”、“./email/ham/23.txt”读取时报编码错误
2.因为rss读取网页错误，　所以rss实例未实现。将一些功能加至classify.py中
错误：rss1 = "http://rss.yule.sohu.com/rss/yuletoutiao.xml"
    rss2 = "http://www.cppblog.com/kevinlynx/category/6337.html/rss"
    tt = feedparser.parse(rss1)
    cb = feedparser.parse(rss2)
    UnicodeEncodeError: 'ascii' codec can't encode characters in position 12-22: ordinal not in range(128)