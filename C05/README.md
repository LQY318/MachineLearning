horse_clic.py  
(1) 收集数据:给定数据文件。data/文件夹中  
(2) 准备数据:用Python解析文本文件并填充缺失值为０。horse_clic.py中的create_data函数生成训练集与验证集  
(3) 训练算法:使用优化算法,找到最佳的系数。logistic_regression.py中的SGD函数  
(4) 测试算法:logistic_regression.py中的classifier_LR函数进行结果预测。为了量化回归的效果,需要观察错误率。约为30%  
(5) 使用算法
