# MIRRec

# Files
The main function is in MyRecEditTrain.py.  Use the 'RecommendByMyRecEdit' for hypergraph-base model training and use the 'EvaluationAlgorithm' to get the evaluation results.

The other .py files are invocated methods for HyperGraph construction.

the .zip files are the preprocessed data of different projects in the dataset, named after the corresponding project.

# Steps

Input: The pre-processed training dataset (zip file).

Source code: MyRecEditTrain.py 

# Runing example：

# parameters setting
    # time interval selection
    dates =[(2018, 6, 2019, 6), (2018, 6, 2019, 7), (2018, 6, 2019, 8), (2018, 6, 2019, 9), (2018, 6, 2019, 10),
                 (2018, 6, 2019, 11), (2018, 6, 2019, 12), (2018, 6, 2020, 1), (2018, 6, 2020, 2), (2018, 6, 2020, 3)]
    #projects selection
    projects = ['electron', 'opencv', 'xbmc', 'react', 'angular', 'django',
                 'symfony', 'rails', 'scala']
    alpha = 0.9
    K = 10
    re_arr = [1, 2, 3, 4, 5]#Top-k
    c = 0.8

# training
    for re in re_arr:
        for p in projects:
              MyRecEditTrain.RecommendByMyRecEdit(train_data, train_data_commit, train_data_issue_comment,
                                       train_data_review_comment, train_data_y, train_data_y_workload,
                                       train_data_committer, train_data_issue_commenter, train_data_review_commenter,
                                       test_data, test_data_commit, test_data_y,
                                       test_data_y_workload, test_data_committer, date,
                                       project, convertDict, recommendNum=5,
                                       K=10, alpha=0.8, c=1,
                                       TrainPRDisIsComputed=False,
                                       HyperGraphIsCreated=False
                                       , re=4, ct=3, ic=1, rc=1)
# Evalution for metrics
    for re in re_arr:
        for p in projects:
            HyperGraphIsCreated = True  # continue training without initialization, or construct the initial hypergraph(False)
            TrainPRDisIsComputed = True  # pr-pr weight updated (for optimal k)
            MyRecEditTrain.TestAlgorithm(p, dates, alpha=alpha, K=K, c=c, TrainPRDisIsComputed=TrainPRDisIsComputed,
                                         HyperGraphIsCreated=HyperGraphIsCreated, re=4, ct=3, ic=1, rc=1)

