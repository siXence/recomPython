import math
import random
import copy
import numpy as np
import time
#coding=utf-8
#__author__ = 'xinxvwin'

def getNowTime():
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))

def hello():
    "test"
    print('Hello XV!')

def str2int_data(data):
    "string数据转为int"
    b = data.split('\t')
    b[-1] = b[-1][0:-1]
    intdata = []
    intdata.append(int(b[0]))
    intdata.append(int(b[1]))
    intdata.append(float(b[2]))
    return intdata

def get_orginal_sample(filepath):
    "由采样的txt得到数据list, 将数据化成user_num*3储蓄，第一列为用户ID，第二类为节目ID，第三类为观看时长比"
    print('start get_orginal_sample...')
    f = open(filepath,'r')
    alllines = f.readlines()
    f.close()
    myLine = []
    for eachline in alllines:
        myLine.append(eachline)
    l = len(myLine)
    original_data = []
    for i in range(l):
        original_data.append(str2int_data(myLine[i]))
    return original_data

def proID2seq(original_data):
    "节目ID重新编号, 返回字典，其中key为节目原来的ID， value为节目ID对应的编号， 编号为1~3000，因为采样的节目数为3000"
    print('start proID2seq...')
    leng = len(original_data)
    proid2seq = {}
    j = 0
    for i in range(leng):
        if original_data[i][1] not in proid2seq.keys():
            proid2seq[original_data[i][1]] = j
            j = j + 1
    return proid2seq

def saveProSeq(proid2seq, filepath):
    "将节目ID对应的序列号保存在文件中"
    print('start saveProSeq...')
    leng = len(proid2seq)
    new_list = [0]*leng;
    for key in proid2seq.keys():
        new_list[proid2seq[key]] = key
    fl=open(filepath, 'w')
    for i in new_list:
        fl.write(str(i))
        fl.write("\n")
    fl.close()


# def get_upmat(original_data, proid2seq):
#     "得到用户和节目的矩阵, 用户数为4000, 返回用户-节目向量，用户和节目都是重新编排的序列号， 其值为观看时长比"
#     leng  = len(original_data)
#     upmat = [[0]*len(proid2seq) for i in range(4000)]
#     for i in range(leng):
#         original_data[i][1] = proid2seq[original_data[i][1]]
#         upmat[original_data[i][0]-1][original_data[i][1]] = original_data[i][2]
#     return upmat

def get_upmat(original_data, proid2seq):
    '''得到用户和节目的矩阵, 用户数为4000, 返回用户-节目矩阵，用户和节目都是重新编排的序列号， 其值为观看时长
        按照4:1的比例，得到训练矩阵和测试矩阵
    '''
    print('start get_upmat...')
    leng  = len(original_data)
    upmat = [[0]*len(proid2seq) for i in range(4000)]
    upmat_train = [[0]*len(proid2seq) for i in range(4000)]  #用于训练
    upmat_test = [[0]*len(proid2seq) for i in range(4000)]  #用于测试
    for i in range(leng):
        original_data[i][1] = proid2seq[original_data[i][1]]
        upmat[original_data[i][0]-1][original_data[i][1]] = original_data[i][2]
        upmat_train[original_data[i][0]-1][original_data[i][1]] = original_data[i][2]
        upmat_test[original_data[i][0]-1][original_data[i][1]] = original_data[i][2]

    # #归一化处理
    # vaild = [0]*len(upmat[0])
    # for i in range(len(upmat[0])):
    #     for j in range(len(upmat)):
    #         vaild[i] += upmat[j][i]
    # for i in range(len(upmat)):
    #     for j in range(len(upmat[0])):
    #         upmat[i][j] = upmat[i][j]/vaild[j]
    #         upmat_train[i][j] = upmat_train[i][j]/vaild[j]
    #         upmat_test[i][j] = upmat_test[i][j]/vaild[j]

    l1 = len(upmat)
    l2 = len(upmat[0])
    for i in range(l1):
        for j in range(l2):
            if upmat[i][j] != 0:
                if random.randint(1,10) > 2:
                    upmat_test[i][j] = 0
                else:
                    upmat_train[i][j] = 0
    fl=open( 'D:\毕设\接收\data\my_upmat_test2.txt', 'w')
    for i in range(l1):
        for j in range(l2):
            fl.write(str(upmat_test[i][j]))
            fl.write(' ')
        fl.write("\n")
    fl.close()
    return [upmat_train,original_data, upmat_test]

def pre_process(filepath):
    '''输入为原始采样数据，用户和节目的矩阵经过了修正，没有经过修正的矩阵,数据的预处理只要调用者一个函数就可以，整合了预处理的函数。
    输出为用户和节目矩阵
    '''
    print('start pre_process...')
    original_data = get_orginal_sample(filepath)
    proid2seq = proID2seq(original_data)
    filepath1 =  'D:\毕设\接收\data\list2.txt'
    saveProSeq(proid2seq, filepath1)
    [upmat,original_data, upmat_test] = get_upmat(original_data, proid2seq)
    #进一步处理，去除大于1.5的时长比，这是属于错误数据，并且做tan3映射,可以参考matlab程序cleandata.m
    m = len(upmat)
    n = len(upmat[0])
    print('m:',m)
    print('n:',n)
    for i in range(m):
        for j in range(n):
            if upmat[i][j] > 1.5 or upmat[i][j] == 0:
                upmat[i][j] = 0.5
            elif upmat[i][j] > 1:
                upmat[i][j] = 1
            upmat[i][j] = math.pow(math.tan(upmat[i][j]*(math.pi/2)-math.pi/4),3)
            # upmat[i][j] = math.pow(math.tan(upmat[i][j]*math.pi/2-math.pi/4),1)
    return [upmat,original_data, proid2seq, upmat_test]

def cal_dist(col_upmat, x, y):
    "计算两点间距离,输入为处理后的节目序号"
    dis = abs(col_upmat[x]-col_upmat[y])
    # n = len(upmat)
    # dis = 0
    # for i in range(n):
    #     dis = dis + abs(upmat[i][x]-upmat[i][y])
    return dis


def get_dismat(upmat):
    "输入为用户-节目矩阵，输出为节目距离矩阵"
    print('start get_dismat...')
    n = len(upmat[0])
    user_num = len(upmat)
    dismat = [[0]*n for i in range(n)]
    mx = np.asarray(upmat).transpose() #转置 变为节目*用户
    for i in range(n):
        print('the ',i,' times...')
        for j in range(i+1,n):
            # dismat[i][j] = cal_dist(col_upmat, i, j)
            dismat[i][j] = sum((abs(mx[i]-mx[j])))
    return dismat

def get_cos_dismat(upmat):
    '''
    余弦距离
    :param upmat:
    :return:
    '''
    print('start get_cos_dismat...')
    n = len(upmat[0])
    user_num = len(upmat)
    dismat = [[0]*n for i in range(n)]
    mx = np.asarray(upmat).transpose() #转置 变为节目*用户
    dismat = np.asarray(dismat)
    for i in range(n):
        print('the ',i,' times...')
        for j in range(i+1,n):
            # dismat[i][j] = cal_dist(col_upmat, i, j)
            if math.sqrt(sum(pow(mx[i],2)))*math.sqrt(sum(pow(mx[j],2))) != 0:
                dismat[i][j] = sum(mx[i]*mx[j])/(math.sqrt(sum(pow(mx[i],2)))*math.sqrt(sum(pow(mx[j],2))))
    return dismat.tolist()


#开始聚类,上面都是对数据的处理部分
def getdis(pos1, pos2, dismat):
    "读取节目之间的距离"
    if pos1>pos2:
        return getdis(pos2, pos1, dismat)
    else:
        return dismat[pos1][pos2]


def clustering(centers, dismat, Clusters, unprotect):
    "最大最小聚类"
    k = len(centers)
    Dis = [0]*k;
    # Dis = []
    L = len(unprotect)
    for i in range(L):
        if unprotect[i] in centers:
            continue
        else:
            for j in range(k):
                Dis[j] = getdis(centers[j], unprotect[i], dismat)
                # Dis.append(getdis(centers[j], unprotect[i], dismat))
            min_index = Dis.index(min(Dis))
            Clusters[3][min_index] = Clusters[3][min_index]+1
            Clusters[2][min_index].append(unprotect[i])
    return Clusters

def add_newcenter(Clusters, class_num, upmat, dismat,clu_min_lim):
    "添加新的中心簇"
    centers = [0]*(class_num+1)
    Max_clu_num = 0
    k = 0
    for i in range(class_num):
        centers[i] = Clusters[0][i]
        if Clusters[3][i] > Max_clu_num:
            Max_clu_num = Clusters[3][i]
            k=i
    Max_clu_pro = Clusters[2][k]
    tmp = [0]*class_num
    maxdis = -1
    for i in range(Max_clu_num):
        for j in range(class_num):
            tmp[j] = getdis(Max_clu_pro[i], Clusters[0][j], dismat)
        if min(tmp)>maxdis:
            maxdis = min(tmp)
            newcenter = Max_clu_pro[i]
    centers[-1] = newcenter
    # class_num = class_num + 1
    # Clusters[0][class_num] = newcenter
    # Clusters[1][class_num] = [upmat[x][newcenter] for x in range(len(upmat))]
    # Clusters[2][class_num] = newcenter
    # Clusters[3][class_num] = 1
    Clusters[0].append(newcenter)
    Clusters[1].append([upmat[x][newcenter] for x in range(len(upmat))])
    Clusters[2].append([newcenter])
    Clusters[3].append(1)
    unprotect = []
    for i in range(class_num):
        if Clusters[3][i]>clu_min_lim:
            unprotect = unprotect + Clusters[2][i]
            Clusters[2][i].clear()
            Clusters[2][i].append(Clusters[0][i])
            Clusters[3][i] = 1
    return [centers, Clusters, unprotect]

# def add_newcenter(Clusters, class_num, upmat, dismat,clu_min_lim):
#     "添加新的中心簇"
#     centers = [0]*(class_num+1)
#     Max_clu_num = 0
#     k = 0
#     if class_num < 4:
#         clu = 1
#     else:
#         clu = 3
#     Max_clu_pro = []
#     for j in range(int(clu)):
#         Max_clu_num = 0
#         k = -1
#         for i in range(class_num):
#             centers[i] = Clusters[0][i]
#             if centers[i] not in Max_clu_pro and Clusters[3][i] > Max_clu_num:
#                 Max_clu_num = Clusters[3][i]
#                 k=i
#         Max_clu_pro += Clusters[2][k]
#         if len(Max_clu_pro) > len(dismat)/10:
#             break
#     tmp = [0]*class_num
#     maxdis = -1
#     for i in range(Max_clu_num):
#         for j in range(class_num):
#             tmp[j] = getdis(Max_clu_pro[i], Clusters[0][j], dismat)
#         if min(tmp)>maxdis:
#             maxdis = min(tmp)
#             newcenter = Max_clu_pro[i]
#     centers[-1] = newcenter
#     # class_num = class_num + 1
#     # Clusters[0][class_num] = newcenter
#     # Clusters[1][class_num] = [upmat[x][newcenter] for x in range(len(upmat))]
#     # Clusters[2][class_num] = newcenter
#     # Clusters[3][class_num] = 1
#     Clusters[0].append(newcenter)
#     Clusters[1].append([upmat[x][newcenter] for x in range(len(upmat))])
#     Clusters[2].append([newcenter])
#     Clusters[3].append(1)
#     unprotect = []
#     for i in range(class_num):
#         if Clusters[3][i]>clu_min_lim:
#             unprotect = unprotect + Clusters[2][i]
#             Clusters[2][i].clear()
#             Clusters[2][i].append(Clusters[0][i])
#             Clusters[3][i] = 1
#     return [centers, Clusters, unprotect]

def get_max_index(data):
    "找到最大元素的索引"
    m = len(data)
    n = len(data[0])
    data_max = -2;
    x = -1
    y = -1
    for i in range(m):
        for j in range(n):
            if data[i][j] > data_max:
                data_max = data[i][j]
                x,y = i,j
    return [x,y]

#def cluster_process(MAXC, dismat, upmat, prolist, clu_min_lim):
def cluster_process(MAXC, dismat, upmat, clu_min_lim):
    "聚类的过程"
    print('start cluster_process...')
    centers = get_max_index(dismat)
    class_num = 2
    Clusters = [[],[],[],[]]
    for i in range(2):
        Clusters[0].append(centers[i])
        Clusters[1].append([upmat[x][centers[i]] for x in range(len(upmat))])
        Clusters[2].append([centers[i]])
        Clusters[3].append(1)
    unprotect = [i for i in range(len(dismat))]
    results = []
    result = []
    while class_num<=MAXC:
        Clusters = clustering(centers, dismat, Clusters, unprotect)
        results.append(Clusters)
        if class_num == MAXC:
            result = Clusters[2]
        # f = open('D:\毕设\接收\data\clu_result.txt','a')
        # if class_num == MAXC-1:
        #     result = Clusters[2]
        #     f = open('D:\毕设\接收\data\clu_result.txt','a')
        #     l1 = len(Clusters[2])
        #     for i in range(l1):
        #         f.write(str(i+1))
        #         f.write('*****************************\n')
        #         for j in range(len(Clusters[2][i])):
        #             f.write(str(Clusters[2][i][j]))
        #             f.write('   ')
        #         f.write('\n')
        #     f.close()
        if class_num < MAXC:
            [centers, Clusters, unprotect] = add_newcenter(Clusters, class_num, upmat, dismat, clu_min_lim)
        class_num = class_num + 1
    print('Clusters[0]:',Clusters[0], ' len:', len(Clusters[0]))
    return [result,Clusters[0]]


# def topn(original_data, MAXC, clu_num):
#     "根据聚类结果得到用户-节目类别矩阵, 用户数为4000， 聚类数为MAXC, 通过new_upmat对每一行降序排列，就可以得到用户喜欢的topN节目,最后的new_upmat保存的信息是每个用户对每一类节目的喜欢度，就是把一个类别的节目的观看时长比加和在了一起"
#     print('start topn...')
#     new_upmat = [[0]*MAXC for i in range(4000)]
#     leng = len(original_data)
#     l2 = len(clu_num)
#     ind = 0
#     # print('original_data:',original_data, ' , leng:',leng)
#     for i in range(leng):
#         for j in range(l2):
#             if original_data[i][1] in clu_num[j]:
#                 ind = j
#                 break
#         # print('original_data[i][0]-1:',original_data[i][0]-1,',ind:',ind)
#         new_upmat[original_data[i][0]-1][ind] += original_data[i][2]
#     return new_upmat


def topn(upmat_train, MAXC, clu_num, new_pcmat):
    "根据聚类结果得到用户-节目类别矩阵, 用户数为4000， 聚类数为MAXC, 通过new_upmat对每一行降序排列，就可以得到用户喜欢的topN节目,最后的new_upmat保存的信息是每个用户对每一类节目的喜欢度，就是把一个类别的节目的观看时长比加和在了一起"
    print('start topn...')
    user_num = len(upmat_train)
    pro_num = len(upmat_train[0])
    ind = 0
    new_upmat = []
    # for i in range(user_num):
    #     print('#######################################',i)
    #     print('l1')
    #     s = [0]*MAXC
    #     for j in range(pro_num):
    #         tmp = [x*upmat_train[i][j] for x in new_pcmat[j]]
    #         s = [s[i]+tmp[i]  for i in range(MAXC)]
    #     res = 0
    #     print('l2')
    #     for i in range(MAXC):
    #         res += pow(s[i],2)
    #     print('l3')
    #     fm = math.sqrt(res)
    #     if fm != 0:
    #         s = [x/fm for x in s]
    #     print('l4')
    #     new_upmat.append(s)
    for i in range(user_num):
        s = [0]*MAXC
        s = np.asarray(s)
        new_pcmat = np.asarray(new_pcmat)
        for j in range(pro_num):
            tmp = upmat_train[i][j]*new_pcmat[j]
            s = s+tmp
        # res = sum(pow(s,2))
        # fm = math.sqrt(res)
        # if fm != 0:
        #     s = s/fm
        s = s.tolist()
        new_upmat.append(s)

    return new_upmat


# def get_topn(new_upmat):
#     "得到用户喜欢节目类别的降序排列"
#     print('start get_topn...')
#     sub_new_upmat = new_upmat
#     n = len(new_upmat)
#     m = len(new_upmat[0])
#     print('start get_topn...1')
#     f = open('D:\毕设\接收\data\ltest2.txt','w')
#     MAXC = 300
#     print('start get_topn...2')
#     de_order_clusters = [[0]*m for i in range(n)]
#     print('start get_topn...3')
#     for i in range(n):
#         # print('start get_topn...4')
#         for j in range(m):
#             tmp = max(sub_new_upmat[i])
#             ind = new_upmat[i].index(tmp)
#             sub_new_upmat[i][ind] = -2*m
#             f.write(str(ind))
#             f.write('   ')
#             de_order_clusters[i][j] = ind
#         f.write('\n')
#     print('start get_topn...5')
#     f.close()
#     return de_order_clusters


# def cal_precision_recall(de_order_clusters, clu_result, test_upmat):
#     '''de_order_clusters 横坐标是用户，纵坐标是用户i喜爱节目的降序排列；
#     clu_result 是最后的聚类结果，每个类别里面有节目编号
#     test_upmat 是用户节目测试数据， 对应的值是观看时长比，没有经过tan变换的数据
#
#     return:
#         users_precision m*n的数据，对应m个用户的topn的准确率，纵坐标是topj的结果
#         users_recall 是对应的召回率
#         allusers_precision 是将users_precision列加和，即所用用户对应的平均准确率
#         allusers_recall 是对应的总的
#     '''
#     print('start cal_precision_recall...')
#     user_num = len(de_order_clusters)
#     cluster_num = len(de_order_clusters[0])
#     # tmp_result = clu_result
#     # for i in range(1,cluster_num):
#     #     tmp_result[i] += tmp_result[i-1]
#     users_precision = [[0]*cluster_num for i in range(user_num)]
#     users_recall = [[0]*cluster_num for i in range(user_num)]
#     pro_num = len(test_upmat[0])
#     cnt_view = [0]*user_num;   #用户观看节目数
#     for i in range(user_num):
#         for j in range(pro_num):
#             if test_upmat[i][j] != 0:
#                 cnt_view[i] += 1
#     for i in range(user_num):
#         all_num_pre = 0 #计算准确率的分母
#         cnt = 0  #交集的节目个数
#         print('**********************',i)
#         for j in range(cluster_num):
#             for m in range(pro_num):
#                 # print('test_upmat[i][m]:',test_upmat[i][m])
#                 if test_upmat[i][m] != 0 and m in clu_result[de_order_clusters[i][j]]:
#                     cnt += 1
#             all_num_pre += len(clu_result[de_order_clusters[i][j]])
#             users_precision[i][j] = cnt/all_num_pre
#             if cnt_view[i] != 0:
#                 users_recall[i][j] = cnt/cnt_view[i]
#     allusers_precision = [0]*cluster_num
#     allusers_recall = [0]*cluster_num
#     for i in range(cluster_num):
#         for j in range(user_num):
#             allusers_precision[i] += users_precision[j][i]
#             allusers_recall[i] += users_recall[j][i]
#         allusers_precision[i] = allusers_precision[i]/user_num
#         allusers_recall[i] = allusers_recall[i]/user_num
#     return [users_precision, users_recall, allusers_precision, allusers_recall]



# def cal_precision_recall(de_order_clusters, clu_result, test_upmat, train_upmat):
#     '''de_order_clusters 横坐标是用户，纵坐标是用户i喜爱节目的降序排列；
#     clu_result 是最后的聚类结果，每个类别里面有节目编号
#     test_upmat 是用户节目测试数据， 对应的值是观看时长比，没有经过tan变换的数据
#
#     return:
#         users_precision m*n的数据，对应m个用户的topn的准确率，纵坐标是topj的结果
#         users_recall 是对应的召回率
#         allusers_precision 是将users_precision列加和，即所用用户对应的平均准确率
#         allusers_recall 是对应的总的
#     '''
#
#     print('start cal_precision_recall...')
#     user_num = len(de_order_clusters)
#     cluster_num = len(de_order_clusters[0])
#     topn = 200
#     user_recommend_topn = [[0]*topn for i in range(user_num)]
#     for i in range(user_num):
#         ind = 0
#         for j in range(cluster_num):
#             len_tmp = len(clu_result[de_order_clusters[i][j]])
#             for m in range(len_tmp):
#                 if train_upmat[i][clu_result[de_order_clusters[i][j]][m]] == 0:
#                     user_recommend_topn[i][ind] = clu_result[de_order_clusters[i][j]][m]
#                     ind += 1
#                     if ind == topn:
#                         break
#             if ind == topn:
#                 break
#     users_precision = [[0]*topn for i in range(user_num)]
#     users_recall = [[0]*topn for i in range(user_num)]
#     pro_num = len(test_upmat[0])
#     cnt_view = [0]*user_num;   #用户观看节目数
#     for i in range(user_num):
#         for j in range(pro_num):
#             if test_upmat[i][j] != 0:
#                 cnt_view[i] += 1
#
#
#     for i in range(user_num):
#         # all_num_pre = 0 #计算准确率的分母
#         # cnt = 0  #交集的节目个数
#         print('**********************',i)
#         for j in range(topn):
#             cnt = 0  #交集的节目个数
#             for m in range(pro_num):
#                 if test_upmat[i][m] != 0 and m in user_recommend_topn[i][:j+1]:
#                     cnt += 1
#             users_precision[i][j] = cnt/(j+1)
#             if cnt_view[i] != 0:
#                 users_recall[i][j] = cnt/cnt_view[i]
#     allusers_precision = [0]*topn
#     allusers_recall = [0]*topn
#     for i in range(topn):
#         for j in range(user_num):
#             allusers_precision[i] += users_precision[j][i]
#             allusers_recall[i] += users_recall[j][i]
#         # allusers_precision[i] = allusers_precision[i]/user_num
#         # allusers_recall[i] = allusers_recall[i]/user_num
#         allusers_precision[i] = allusers_precision[i]/4000
#         allusers_recall[i] = allusers_recall[i]/4000
#     return [users_precision, users_recall, allusers_precision, allusers_recall]


def get_set(upmat_test):
    '''
    得到用户观看的节目数，用于测试
    :param upmat_test: 用户测试矩阵， 参数为upmat_test，返回testset， 参数为upmat_train， 返回trainset
    :return:用户*节目矩阵，0表示没有观看，1表示观看
    '''
    print('start get_set...')
    user_num = len(upmat_test)
    pro_num = len(upmat_test[0])
    testset = [[0]*pro_num for i in range(user_num)]
    for i in range(user_num):
        for j in range(pro_num):
            if upmat_test[i][j] != 0:
                testset[i][j] = 1
    return testset

def get_new_pcmat(result, pro_num):
    '''

    :param result: 聚类结果
    :param pro_num: 数据中节目的个数
    :return:3000*200矩阵，表明节目归属哪一个聚类
    '''
    print('start get_new_pcmat...')
    clu_num = len(result)
    new_pcmat = [[0]*clu_num for i in range(pro_num)]
    for i in range(pro_num):
        for j in range(clu_num):
            if i in result[j]:
                new_pcmat[i][j] = 1
                break
    return new_pcmat

def matrixMul(A, B):
    '''
    矩阵乘法
    :param A:
    :param B:
    :return:结果矩阵
    '''
    # print('start matrixMul...')
    # res = [[0] * len(B[0]) for i in range(len(A))]
    # for i in range(len(A)):
    #     print('matrix****************',i)
    #     for j in range(len(B[0])):
    #         for k in range(len(B)):
    #             res[i][j] += A[i][k] * B[k][j]
    mx = np.matrix(A)
    my = np.matrix(B)
    res = (mx*my).tolist()
    return res

def get_UPrec(new_upmat, new_pcmat):
    '''
    得到用户*节目的预估矩阵，数值为预估的喜爱度
    :param clu_result: 聚类结果
    :param new_upmat: 4000*200矩阵，用户对每一类节目的喜爱度
    :param new_pcmat: 3000*200 节目归属哪一类的矩阵
    :return:用户对节目喜爱度的预估矩阵
    '''
    # user_num = len(new_upmat)
    # pro_num = len(new_pcmat)
    # clu_num = len(new_pcmat[0])
    print('start get_UPrec...')
    new_pcmat = [[r[col] for r in new_pcmat] for col in range(len(new_pcmat[0]))] #转置
    UPrec = matrixMul(new_upmat,new_pcmat)
    return UPrec


def uptest(UPres, trainset, testset):
    '''
    计算准确率和召回率
    :param UPres: 用户*节目预估矩阵
    :param trainset: 用户*节目训练矩阵，为0-1矩阵，1表示观看过结果
    :param testset: 用户*节目训练矩阵，为0-1矩阵，1表示观看过结果
    :return:返回准确率和召回率
    '''
    print('start uptest...')
    user_num = len(UPres)
    topN = 100
    test_seen = [0]*user_num #每个用户观看的节目数
    for i in range(len(testset)):
        for j in range(len(testset[0])):
            test_seen[i] += testset[i][j]

    PRE = [0]*topN
    RECALL = [0]*topN
    for topn in range(1,topN+1):
        print('************************',topn)
        preci = [0]*user_num
        recall = [0]*user_num
        for theOrderOfUser in range(user_num):
            # print('test',1)
            tmp = [UPres[theOrderOfUser], trainset[theOrderOfUser], testset[theOrderOfUser]]
            # print('test',2)
            mx = np.matrix(tmp)
            # print('test',3)
            tmp = mx.transpose().tolist()
            # print('test',4)
            # tmp = [[r[col] for r in tmp] for col in range(len(tmp[0]))]
            tmp.sort(key=lambda x:x[0], reverse=True)
            # print('test',5)
            i = 0
            while i < topn+1:
                if tmp[i][1] != 0:
                    del tmp[i]
                else:
                    i += 1
            cnt = 0
            # print('test',6)
            for i in range(topn):
                if tmp[i][2] == 1:
                    cnt += 1
            # print('test',7)
            preci[theOrderOfUser] = cnt/topn
            if test_seen[theOrderOfUser] != 0:
                recall[theOrderOfUser] = cnt/test_seen[theOrderOfUser]
        PRE[topn-1] = sum(preci)/user_num
        RECALL[topn-1] = sum(recall)/user_num
    return [PRE, RECALL]





#FCM相关函数
def addr(a, strsort):
    '''
    返回向量升序或降序排列后各分量在原始向量中的索引
    :param a:
    :param strsort: 1是降序，0是升序
    :return:
    '''
    if strsort == 1:
        f = np.argsort(-a)
    else:
        f = np.argsort(a)
    return f


def maxrowf(U):
    '''
    求矩阵U每列第一大元素所在行
    :param U:
    :return:
    '''
    N = U.shape[1]
    mr = np.zeros([1,N])
    mr = [0]*N
    for i in range(N):
        aj = addr(U[:,i], 1)
        mr[i] = aj[0]
    return mr



def fcm_clusting(data, C, M, epsm):
    '''
    模糊c均值聚类
    :param data:用户*节目矩阵，输入为训练矩阵 S*N， S为用户数，N为节目数
    :param C: 聚类数，目前选定为200
    :param M: 加权指数，设定为2
    :param epsm: 算法的迭代停止阈值，缺省设定为1.0e-6
    :return:
        U: C*N 型矩阵， FCM的划分矩阵，即每个节目归属值，这个才是我想要的，可以得到每个节目属于的类别
        P: C*S 型矩阵
        Dist:
        Cluster_Res:聚类*节目向量
        Cluster_Res_2:横坐标是聚类，每一行包含的是该类的节目编号，等同于最大最小聚类中返回的result
        iter: FCM迭代次数
    '''
    m = 2/(M-1)
    iter = 0
    Data = np.transpose(np.asarray(data))
    [N, S] = Data.shape
    Dist = np.zeros([C,N])
    U = np.zeros([C,N])
    P = np.zeros([C,S])
    #随机初始化划分矩阵
    U0 = np.random.random([C,N])
    tmp_sum = sum(U0)
    tmp_sum.shape = (tmp_sum.shape[0], 1)
    tmp_sum = np.transpose(tmp_sum)
    tmp2 = np.ones(C)
    tmp2.shape = (tmp2.shape[0],1)
    U0 = U0/(np.dot(tmp2,tmp_sum))
    #FCM的迭代算法
    while True:
        iter += 1
        if iter > 200:
            break
        Um = pow(U0,M)
        Um_1 = sum(np.transpose(Um))
        Um_1.shape = (Um_1.shape[0],1)
        Um_1 = np.transpose(Um_1)
        tmp2 = np.ones(S)
        tmp2.shape = (tmp2.shape[0],1)
        Um_1 = np.dot(tmp2, Um_1)
        fm = np.transpose(Um_1)
        P = np.dot(Um, Data)/fm
        for i in range(C):
            for j in range(N):
                Dist[i][j] = np.linalg.norm(P[i]-Data[j])
        tmp = sum(pow(Dist,-m))
        tmp.shape = (tmp.shape[0],1)
        tmp = np.transpose(tmp)
        tmp2 = np.ones(C)
        tmp2.shape = (tmp2.shape[0],1)
        tmp = np.dot(tmp2, tmp)
        tmp = pow(Dist,m)*tmp
        U = 1/tmp
        if max(sum(abs(np.transpose(U-U0)))) < epsm:
            break
        U0 = U
    print('iter:',iter)
    #聚类结果
    res = maxrowf(U)
    Cluster_Res = [[0]*N for i in range(C)]
    Cluster_Res_2 = [[] for i in range(C)]
    for i in range(N):
        Cluster_Res[res[i]][i] = 1
        Cluster_Res_2[res[i]].append(i)
    return [Cluster_Res, Cluster_Res_2]



def my_maxminclustart(dismat, MAXC):
    '''
    最大最小距离，先找到中心点，再聚类
    :param dismat: 距离矩阵
    :param MAXC: 聚类数
    :return:
    '''
    print('start my_maxminclustart...')
    centers = get_max_index(dismat)
    class_num = 2
    pro_num = len(dismat)
    while class_num < MAXC:
        pro_dis = [0]*pro_num
        for i in range(pro_num):
            if i not in centers:
                tmp = [0]*class_num
                for j in range(class_num):
                    tmp[j] = getdis(i,centers[j], dismat)
                pro_dis[i] = min(tmp)
        centers.append(pro_dis.index(max(pro_dis)))
        class_num += 1

    #开始聚类
    Dis = [0]*MAXC
    result = [[] for i in range(MAXC)]
    for i in range(MAXC):
        result[i].append(centers[i])
    for i in range(pro_num):
        if i in centers:
            continue
        else:
            for j in range(MAXC):
                Dis[j] = getdis(centers[j], i, dismat)
            min_index = Dis.index(min(Dis))
            result[min_index].append(i)
    resul_2 = my_maxminclustart2(dismat, MAXC+1, result[-1])
    del result[-1]
    result += resul_2
    return result


def my_maxminclustart2(dismat, MAXC, data):
    '''
    最大最小距离，先找到中心点，再聚类
    :param dismat: 距离矩阵
    :param MAXC: 聚类数
    :return:
    '''
    print('start my_maxminclustart...')
    max_dis = 0
    node1 = -1
    node2 = -1
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            dis = getdis(data[i], data[j], dismat)
            if dis > max_dis:
                node1 = data[i]
                node2 = data[j]
    centers = [node1, node2]
    class_num = 2
    pro_num = len(data)
    while class_num < MAXC:
        pro_dis = [0]*pro_num
        for i in range(pro_num):
            if data[i] not in centers:
                tmp = [0]*class_num
                for j in range(class_num):
                    tmp[j] = getdis(data[i],centers[j], dismat)
                pro_dis[i] = min(tmp)
        centers.append(data[pro_dis.index(max(pro_dis))])
        class_num += 1

    #开始聚类
    Dis = [0]*MAXC
    result = [[] for i in range(MAXC)]
    for i in range(MAXC):
        result[i].append(centers[i])
    for i in range(pro_num):
        if data[i] in centers:
            continue
        else:
            for j in range(MAXC):
                Dis[j] = getdis(centers[j], data[i], dismat)
            min_index = Dis.index(min(Dis))
            result[min_index].append(data[i])

    return result

def re_cluster(result, MAXC, dismat):
    '''
    对最大的类重新聚类
    :param result:
    :param MAXC:
    :param dismat:
    :return:
    '''
    num = 0
    max_ind = 0
    for i in range(len(result)):
        if num < len(result[i]):
            num = len(result[i])
            max_ind = i
    max_clu = result[max_ind]
    del result[max_ind]




####################################################多中心聚类算法

def average_centers(centers, m, dismat):
    '''

    :param centers:
    :param m:
    :return:
    '''
    dis = 0
    class_num = len(centers)
    for i in range(class_num):
        for j in range(i+1, class_num):
            dis += getdis(centers[j], centers[i], dismat)
    if class_num > 1:
        dis = dis/(class_num*(class_num-1)/2)
    return m*dis


def multi_centers_clustering(centers, dismat):
    "最大最小聚类"
    # print('centers:', centers)
    # print('len(center):', len(centers))

    k = len(centers)
    Dis = [0]*k
    # Dis = []
    L = len(dismat)
    clu_res = [[] for i in range(k)]
    for i in range(L):
        if i in centers:
            continue
        else:
            for j in range(k):
                Dis[j] = getdis(centers[j], i, dismat)
                # Dis.append(getdis(centers[j], unprotect[i], dismat))
            min_index = Dis.index(min(Dis))
            clu_res[min_index].append(i)
    return clu_res

def multi_centers_add_newcenter(clu_res, dismat, centers, m):
    "添加新的中心簇"
    class_num = len(centers)
    # new_centers = [0]*(class_num+1)
    Max_clu_num = 0
    k = 0
    for i in range(class_num):
        if len(clu_res[i]) > Max_clu_num:
            Max_clu_num = len(clu_res[i])
            k=i
    Max_clu_pro = clu_res[k]
    tmp = [0]*class_num
    maxdis = -1
    for i in range(Max_clu_num):
        for j in range(class_num):
            tmp[j] = getdis(Max_clu_pro[i], centers[j], dismat)
        if min(tmp)>maxdis:
            maxdis = min(tmp)
            newcenter = Max_clu_pro[i]
    if len(centers) > 1 and maxdis <= average_centers(centers, m, dismat):
        newcenter = -1
        return newcenter
    # for i in range(class_num):
    #     clu_res[i] = [centers[i]]
    return newcenter








def multi_centers_max_min_cluster(dismat, multi_num = 10, m = 0.2):
    '''
    多中心聚类
    :param upmat_train:
    :param dismat:
    :param multi_num: 中心点的组数，默认值是10
    :param m: 距离计算参数,默认值为0.5
    :return:
    '''

    pro_num = len(dismat)
    centers = [[] for i in range(multi_num)]
    centers[0] = get_max_index(dismat)
    #第一组，选定最远的两个点最为初始簇中心
    print('centers[0]:', centers[0])
    iter = 1
    while True:
        # newCenter = -1
        # for i in range(pro_num):
        #     if i in centers[0]:
        #         continue
        #     for j in range(len(centers[0])):
        clu_res = multi_centers_clustering(centers[0], dismat)
        newcenter = multi_centers_add_newcenter(clu_res, dismat, centers[0], m)
        if newcenter == -1:
            break
        centers[0].append(newcenter)
        iter += 1
        if iter > 200:
            del centers[0][-1]
            break
    print('certers[0]:',centers[0])


    # for i in range(1, multi_num):
    #     centers[i] = [random.randint(0,pro_num-1)]
    #     iter = 1
    #     while True:
    #         clu_res = multi_centers_clustering(centers[i], dismat)
    #         newcenter = multi_centers_add_newcenter(clu_res, dismat, centers[i], m)
    #         if newcenter == -1:
    #             break
    #         iter += 1
    #         centers[i].append(newcenter)
    #         if iter > 300:
    #             break
    #     print('certers[',i,']:',centers[i])
    # all_centers = []
    # for i in range(multi_num):
    #     all_centers = list(set(all_centers).union(set(centers[i])))
    # max_dis = -1
    # start_node = -1
    # end_node = -1
    # for i in range(len(all_centers)):
    #     for j in range(i+1,len(all_centers)):
    #         dis = getdis(all_centers[i], all_centers[j], dismat)
    #         if dis > max_dis:
    #             max_dis = dis
    #             start_node = all_centers[i]
    #             end_node = all_centers[j]
    # final_centers = [start_node, end_node]
    # iter = 1
    # while True:
    #     # clu_res = multi_centers_clustering(final_centers, dismat)
    #     # [final_centers, newcenter] = multi_centers_add_newcenter(clu_res, dismat, final_centers, m)
    #     # if newcenter == -1:
    #     #     break
    #     # iter += 1
    #     # if iter >= 400:
    #     #     break
    #     Dis = [0]*len(final_centers)
    #     maxdis = -1
    #     newcenter = -1
    #     for i in range(len(all_centers)):
    #         if all_centers[i] in final_centers:
    #             continue
    #         for j in range(len(final_centers)):
    #             Dis[j] = getdis(all_centers[i], final_centers[j], dismat)
    #         if min(Dis) > maxdis:
    #              maxdis = min(Dis)
    #              newcenter = all_centers[i]
    #     if len(final_centers) > 1 and maxdis > average_centers(final_centers, m, dismat):
    #         final_centers.append(newcenter)
    #     else:
    #         break
    #     iter += 1
    #     if iter > 300:
    #         break
    # result = multi_centers_clustering(final_centers, dismat)
    # return result
    return [clu_res, centers[0]]








####################################################################################将节目归属模糊化
def get_pcmat(centers, pro_num, dismat):
    '''
    得到节目*聚类矩阵
    算出每个节目含有的隐因子的比例
    :param centers:
    :param pro_num:
    :return:
    '''
    clu_num = len(centers)
    print('clu_num:', clu_num)
    pcmat = [[0.0]*clu_num for i in range(pro_num)]
    pcmat = np.asarray(pcmat)
    for i in range(pro_num):
        if i in centers:
            pcmat[i][centers.index(i)] = 1
        else:
            for j in range(clu_num):
                # print('1/getdis(i, centers[j], dismat):', 100/getdis(i, centers[j], dismat))
                pcmat[i][j] = 100/getdis(i, centers[j], dismat)
            # tmp = sum(pow(pcmat[i], 2))
            # # print('tmp:',tmp)
            # # print('befor pcmat[i]:',pcmat[i])
            # pcmat[i] = pow(pcmat[i], 2)/tmp
            # # print('after pcmat[i]:',pcmat[i])
            # # print('after sum:', sum(pcmat[i]))


            tmp = math.sqrt(sum(pow(pcmat[i], 2)))
            # print('tmp:',tmp)
            # print('befor pcmat[i]:',pcmat[i])
            pcmat[i] = pcmat[i]/tmp
            # print('after pcmat[i]:',pcmat[i])
            # print('after sum:', sum(pcmat[i]))

            # tmp = sum(pcmat[i])
            # pcmat[i] = pcmat[i]/tmp


    # Dis = [0.0]*clu_num
    # for i in range(pro_num):
    #     if i in centers:
    #         pcmat[i][centers.index(i)] = 1
    #         continue
    #     for j in range(clu_num):
    #         Dis[j] = getdis(i, centers[j], dismat)
    #     dis = np.asarray(Dis)
    #     dis = pow(dis,2)
    #     for j in range(clu_num):
    #         pcmat[i][j] = 1/sum(dis[j]/dis)
    #     print('pcmat[i]:',pcmat[i], ' sum pcmat:', sum(pcmat[i]))

    return pcmat.tolist()


def get_ucmat(upmat, pcmat):
    clu_num = len(pcmat[0])
    user_num = len(upmat)
    ucmat = [[0.0]*clu_num for i in range(user_num)]
    ucmat = np.asarray(ucmat)
    upmat = np.asarray(upmat)
    pcmat = np.transpose(np.asarray(pcmat))
    for i in range(user_num):
        # print('get_ucmatget_ucmat##############',i)
        for j in range(clu_num):
            ucmat[i][j] = sum(upmat[i]*pcmat[j])
        # print('ucmat[i]:', ucmat[i])
        tmp = pow(ucmat[i],2)
        tmp = sum(tmp)
        tmp = math.sqrt(tmp)
        print('ucmat tmp:',tmp)
        ucmat[i] = ucmat[i]/tmp
        # print('ucmat[i]:',ucmat[i])
    return ucmat.tolist()




################################################调整中心的聚类
def adjust_centers_clusters(dismat, upmat, MAXC):
    '''

    :param dismat:
    :param user_num:
    :return:
    '''
    print('start adjust_centers_clusters...')
    pro_num = len(dismat)
    user_num = len(upmat)
    node = get_max_index(dismat)
    centers = [[0.0]*MAXC for i in range(user_num)]
    centers = np.asarray(centers)
    upmat = np.asarray(upmat)
    centers[:,0] = upmat[:,node[0]]
    centers[:,1] = upmat[:,node[1]]
    iter = 2
    result = []
    while iter < MAXC:
        print('iter:',iter)
        clu_res = [[] for i in range(iter)]
        Dis = [0.0]*iter
        maxdis = -1
        max_ind = -1
        for i in range(pro_num):
            for j in range(iter):
                Dis[j] = sum(abs(upmat[:,i]-centers[:,j]))
            clu_res[Dis.index((min(Dis)))].append(i)

        for i in range(iter):
            print('clu_res[',i,']:',clu_res[i])

        for i in range(iter):
            # print('clu_res[i]:',clu_res[i])
            l = len(clu_res[i])
            if l > 0:
                centers[:,i] = upmat[:,clu_res[i][0]]
                for j in range(1,l):
                    centers[:,i] += upmat[:,clu_res[i][j]]
                centers[:,i] = centers[:,i]/l
                print('centers:',centers[:,i].tolist())

        iter += 1
        if iter < MAXC:
            max_group = -1
            max_num = -1
            for i in range(iter-1):
                if len(clu_res[i]) > max_num:
                    max_group = i
                    max_num = len(clu_res[i])
            maxdis = -1
            max_ind = -1
            for i in range(len(clu_res[max_group])):
                tmp = sum(abs(upmat[:,clu_res[max_group][i]]-centers[:,max_group]))
                if tmp > maxdis:
                    maxdis = tmp
                    max_ind = clu_res[max_group][i]
            centers[:,iter-1] = upmat[:,max_ind]

    # print('clu_res:',clu_res[199])
    return  clu_res









##########################################################################分裂层次聚类,效果类似，但是速度更快

def get_max_index_in(data, dismat):
    "找到最大元素的索引"
    m = len(data)
    data_max = -2
    x = -1
    y = -1
    for i in range(m):
        for j in range(i+1,m):
            if getdis(data[i], data[j], dismat) > data_max:
                data_max = getdis(data[i], data[j], dismat)
                x,y = data[i],data[j]
    return [x,y]




def desprade_cluster(dismat, MAXC):

    pro_num = len(dismat)
    data = [i for i in range(pro_num)]
    centers = get_max_index(dismat)
    test = get_max_index_in(data, dismat)
    print('test my method-----------------------', centers, '  and  ', test)
    nodes = centers
    clu_res = []
    #第一组，选定最远的两个点最为初始簇中心
    print('centers[0]:', centers)
    class_num = 2
    while True:
        print('class_num:', class_num)
        clu_res.append([nodes[0]])
        clu_res.append([nodes[1]])
        l = len(data)
        Dis = [0.0]*2
        for i in range(l):
            if data[i] in nodes:
                continue
            for j in range(2):
                Dis[j] = getdis(data[i], nodes[j], dismat)
            if Dis[0] > Dis[1]:
                clu_res[class_num-1].append(data[i])
            else:
                clu_res[class_num-2].append(data[i])

        if class_num == MAXC:
            break

        max_clu_num = -1
        max_clu_ind = -1
        for i in range(class_num):
            if len(clu_res[i]) > max_clu_num:
                max_clu_num = len(clu_res[i])
                max_clu_ind = i
        data = clu_res[max_clu_ind]
        del clu_res[max_clu_ind]
        nodes = get_max_index_in(data, dismat)

        class_num += 1



    return clu_res







def desprade_cluster2(dismat, MAXC):

    pro_num = len(dismat)
    data = [i for i in range(pro_num)]
    centers = get_max_index(dismat)
    test = get_max_index_in(data, dismat)
    print('test my method-----------------------', centers, '  and  ', test)
    nodes = centers
    clu_res = []
    #第一组，选定最远的两个点最为初始簇中心
    print('centers[0]:', centers)
    class_num = 2
    while True:
        print('class_num:', class_num)
        clu_res = []
        for i in range(class_num):
            clu_res.append([centers[i]])
        # l = len(data)
        Dis = [0.0]*class_num
        for i in range(pro_num):
            if data[i] in centers:
                continue
            for j in range(len(centers)):
                Dis[j] = getdis(data[i], centers[j], dismat)
            clu_res[Dis.index(min(Dis))].append(data[i])

        if class_num == MAXC:
            break

        max_clu_num = -1
        max_clu_ind = -1
        for i in range(class_num):
            if len(clu_res[i]) > max_clu_num:
                max_clu_num = len(clu_res[i])
                max_clu_ind = i
        del centers[max_clu_ind]
        nodes = get_max_index_in(clu_res[max_clu_ind], dismat)
        centers += nodes
        class_num += 1


    print('clu_res:',clu_res)
    return clu_res







def desprade_cluster3(dismat, MAXC):

    pro_num = len(dismat)
    data = [i for i in range(pro_num)]
    centers = get_max_index(dismat)
    test = get_max_index_in(data, dismat)
    print('test my method-----------------------', centers, '  and  ', test)
    nodes = [0.0]*2
    nodes[0] = centers[0]
    nodes[1] = centers[1]
    clu_res = []
    #第一组，选定最远的两个点最为初始簇中心
    print('centers[0]:', centers)
    class_num = 2
    while True:
        print('class_num:', class_num)
        clu_res.append([nodes[0]])
        clu_res.append([nodes[1]])
        l = len(data)
        Dis = [0.0]*2
        for i in range(l):
            if data[i] in nodes:
                continue
            for j in range(2):
                Dis[j] = getdis(data[i], nodes[j], dismat)
            if Dis[0] > Dis[1]:
                clu_res[class_num-1].append(data[i])
            else:
                clu_res[class_num-2].append(data[i])

        if class_num == MAXC:
            break

        max_clu_num = -1
        max_clu_ind = -1
        for i in range(class_num):
            if len(clu_res[i]) > max_clu_num:
                max_clu_num = len(clu_res[i])
                max_clu_ind = i
        # print('nodes:',nodes, 'max_clu_ind:',max_clu_ind, ' len(cen):',len(centers))
        # a = centers[max_clu_ind]
        # b = nodes[0]
        nodes[0] = centers[max_clu_ind]
        del centers[max_clu_ind]
        data = clu_res[max_clu_ind]
        del clu_res[max_clu_ind]
        # nodes = get_max_index_in(data, dismat)
        ll = len(data)
        DisMean = [0.0]*ll
        for i in range(ll):
            if data[i] == nodes[0]:
                continue
            for j in range(ll):
                DisMean[i] += getdis(data[i], data[j], dismat)
        new_cen = DisMean.index(max(DisMean))
        # a = data[new_cen]
        # print('nodes:', nodes)
        nodes[1] = data[new_cen]
        del data[new_cen]
        centers += nodes

        class_num += 1



    return clu_res



def desprade_cluster_min(dismat, MAXC , clu_num):
    '''
    聚类后合并小类
    :param dismat:
    :param MAXC:
    :return:
    '''

    pro_num = len(dismat)
    data = [i for i in range(pro_num)]
    centers = get_max_index(dismat)
    test = get_max_index_in(data, dismat)
    print('test my method-----------------------', centers, '  and  ', test)
    nodes = centers
    clu_res = []
    #第一组，选定最远的两个点最为初始簇中心
    print('centers[0]:', centers)
    class_num = 2
    while True:
        print('class_num:', class_num)
        clu_res.append([nodes[0]])
        clu_res.append([nodes[1]])
        l = len(data)
        Dis = [0.0]*2
        for i in range(l):
            if data[i] in nodes:
                continue
            for j in range(2):
                Dis[j] = getdis(data[i], nodes[j], dismat)
            if Dis[0] > Dis[1]:
                clu_res[class_num-1].append(data[i])
            else:
                clu_res[class_num-2].append(data[i])

        if class_num == MAXC:
            break

        max_clu_num = -1
        max_clu_ind = -1
        for i in range(class_num):
            if len(clu_res[i]) > max_clu_num:
                max_clu_num = len(clu_res[i])
                max_clu_ind = i
        data = clu_res[max_clu_ind]
        del clu_res[max_clu_ind]
        del centers[max_clu_ind]
        nodes = get_max_index_in(data, dismat)
        centers += nodes

        class_num += 1

    minData = []
    class_num = 0
    while class_num < len(clu_res):
        if len(clu_res[class_num]) <= clu_num:
            minData += clu_res[class_num]
            del centers[class_num]
            del clu_res[class_num]
        else :
            class_num += 1

    leng = len(minData)
    Dis = [0.0]*len(centers)
    for i in range(leng):
        for j in range(len(centers)):
            Dis[j] = getdis(minData[i], centers[j], dismat)
        clu_res[Dis.index(min(Dis))].append(minData[i])
    f = open('D:\毕设\接收\data\cnt1.txt','w')
    for i in range(len(clu_res)):
        # print('len(check_result[i]):',len(check_result[i]))
        f.write(str(len(clu_res[i])))
        f.write('\n')
    f.close()

    return clu_res



############################################## SVD ########################################################################



#calculate the overall average
def Average(upmat):
    '''

    :param upmat:
    :return:
    '''
    row = len(upmat)
    col = len(upmat[0])
    avg = 0.0
    cnt = 0
    for i in range(row):
        for j in range(col):
            if upmat[i][j] != 0:
                cnt += 1
            avg += upmat[i][j]
    return (avg/cnt)


def InerProduct(v1, v2):
    '''向量内积'''
    # result = 0
    # for i in range(len(v1)):
    #     result += v1[i] * v2[i]
    v11 = np.array(v1)
    v22 = np.array(v2)
    return np.dot(v11,v22)


def PredictScore(av, bu, bi, pu, qi):
    pScore = av + bu + bi + InerProduct(pu, qi)
    # if pScore < -1:
    #     pScore = -1
    # elif pScore > 1:
    #     pScore = 1

    return pScore


#def SVD(configureFile, testDataFile, trainDataFile, modelSaveFile, upmat, factorNum, learnRate, regularization ):
def SVD(upmat, factorNum, learnRate = 0.005, regularization = 0.02):
    #get the configure
    averageScore = Average(upmat)
    userNum = len(upmat)
    itemNum = len(upmat[0])
    bi = [0.0 for i in range(itemNum)]
    bu = [0.0 for i in range(userNum)]
    temp = math.sqrt(factorNum)
    # qi = [[(0.1 * random.random() / temp) for j in range(factorNum)] for i in range(itemNum)]
    # pu = [[(0.1 * random.random() / temp)  for j in range(factorNum)] for i in range(userNum)]
    qi = [[random.uniform(-1,1) for j in range(factorNum)] for i in range(itemNum)]
    pu = [[random.uniform(0,1)   for j in range(factorNum)] for i in range(userNum)]
    qi = np.array(qi)
    pu = np.array(pu)
    upmat = np.array(upmat)
    print("initialization end\nstart training\n")

    outMatrix = [[0.0 for j in range(itemNum)] for i in range(userNum)]
    print("userNum = ", userNum, " ,iterNum = ", itemNum)
    #train model
    preRmse = 1000000.0
    for step in range(50):
        for uid in range(userNum):
            # print("uid = ", uid)
            for iid in range(itemNum):
                score = upmat[uid][iid]
                prediction = averageScore + bu[uid] + bi[iid] + np.dot(pu[uid],qi[iid])
                # prediction = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
                eui = score - prediction
                #update parameters
                bu[uid] += learnRate * (eui - regularization * bu[uid])
                bi[iid] += learnRate * (eui - regularization * bi[iid])
                #print("iid = ", iid)
                tmp = copy.deepcopy(pu[uid])
                pu[uid] += learnRate*(eui*qi[iid] - regularization*pu[uid])
                qi[iid] += learnRate*(eui*tmp - regularization*qi[iid])
                # for k in range(factorNum):
                #     temp = pu[uid][k]   #attention here, must save the value of pu before updating
                #     pu[uid][k] += learnRate * (eui * qi[iid][k] - regularization * pu[uid][k])
                #     qi[iid][k] += learnRate * (eui * temp - regularization * qi[iid][k])
        learnRate *= 0.9
        curRmse = Validate(upmat, averageScore, bu, bi, pu, qi, userNum, itemNum)
        print("test_RMSE in step %d: %f" %(step, curRmse))
        if curRmse >= preRmse:
            break
        else:
            preRmse = curRmse

    #write the model to files
    for uid in range(userNum):
        for iid in range(itemNum):
            outMatrix[uid][iid] = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
    print("model generation over")
    return outMatrix


# def SVD(upmat, factorNum, learnRate = 0.01, regularization = 0.05):
#     #get the configure
#     averageScore = Average(upmat)
#     userNum = len(upmat)
#     itemNum = len(upmat[0])
#     bi = [0.0 for i in range(itemNum)]
#     bu = [0.0 for i in range(userNum)]
#     temp = math.sqrt(factorNum)
#     # qi = [[(0.1 * random.random() / temp) for j in range(factorNum)] for i in range(itemNum)]
#     # pu = [[(0.1 * random.random() / temp)  for j in range(factorNum)] for i in range(userNum)]
#     qi = [[random.uniform(-1,1) for j in range(factorNum)] for i in range(itemNum)]
#     pu = [[random.uniform(0,1)   for j in range(factorNum)] for i in range(userNum)]
#     print("initialization end\nstart training\n")
#
#     outMatrix = [[0.0 for j in range(itemNum)] for i in range(userNum)]
#     print("userNum = ", userNum, " ,iterNum = ", itemNum)
#     #train model
#     preRmse = 1000000.0
#     for step in range(100):
#         for uid in range(userNum):
#             print("uid = ", uid)
#             for iid in range(itemNum):
#                 score = upmat[uid][iid]
#                 v1 = np.array(pu[uid])
#                 v2 = np.array(qi[iid])
#                 prediction = averageScore + bu[uid] + bi[iid] + np.dot(v1,v2)
#                 # prediction = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
#                 eui = score - prediction
#                 #update parameters
#                 bu[uid] += learnRate * (eui - regularization * bu[uid])
#                 bi[iid] += learnRate * (eui - regularization * bi[iid])
#                 #print("iid = ", iid)
#                 t1 = np.array(pu[uid])
#                 t2 = np.array(qi[iid])
#                 tmp = copy.deepcopy(t1)
#                 t1 += learnRate*(eui*t2 - regularization*t1)
#                 t2 += learnRate*(eui*tmp regularization*t2)
#
#                 for k in range(factorNum):
#                     temp = pu[uid][k]   #attention here, must save the value of pu before updating
#                     pu[uid][k] += learnRate * (eui * qi[iid][k] - regularization * pu[uid][k])
#                     qi[iid][k] += learnRate * (eui * temp - regularization * qi[iid][k])
#         learnRate *= 0.9
#         curRmse = Validate(upmat, averageScore, bu, bi, pu, qi, userNum, itemNum)
#         print("test_RMSE in step %d: %f" %(step, curRmse))
#         if curRmse >= preRmse:
#             break
#         else:
#             preRmse = curRmse
#
#     #write the model to files
#     for uid in range(userNum):
#         for iid in range(itemNum):
#             outMatrix[uid][iid] = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
#     print("model generation over")
#     return outMatrix

#validate the model
def Validate(upmat, av, bu, bi, pu, qi, userNum, itemNum):
    cnt = 0
    rmse = 0.0
    for uid in range(userNum):
        for iid in range(itemNum):
            cnt += 1
            pScore = PredictScore(av, bu[uid], bi[iid], pu[uid], qi[iid])
            tScore = upmat[uid][iid]
            rmse += (tScore - pScore) * (tScore - pScore)
    return math.sqrt(rmse / cnt)





# #use the model to make predict
# def Predict(configureFile, modelSaveFile, testDataFile, resultSaveFile):
#     #get parameter
#     fi = open(configureFile, 'r')
#     line = fi.readline()
#     arr = line.split()
#     averageScore = float(arr[0].strip())
#     fi.close()
#
#     #get model
#     fi = file(modelSaveFile, 'rb')
#     bu = pickle.load(fi)
#     bi = pickle.load(fi)
#     qi = pickle.load(fi)
#     pu = pickle.load(fi)
#     fi.close()
#
#     #predict
#     fi = open(testDataFile, 'r')
#     fo = open(resultSaveFile, 'w')
#     for line in fi:
#         arr = line.split()
#         uid = int(arr[0].strip()) - 1
#         iid = int(arr[1].strip()) - 1
#         pScore = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
#         fo.write("%f\n" %pScore)
#     fi.close()
#     fo.close()
#     print("predict over")


# if __name__ == '__main__':
#     configureFile = 'svd.conf'
#     trainDataFile = 'ml_data\\training.txt'
#     testDataFile = 'ml_data\\test.txt'
#     modelSaveFile = 'svd_model.pkl'
#     resultSaveFile = 'prediction'
#
#     #print("%f" %Average("ua.base"))
#     SVD(configureFile, testDataFile, trainDataFile, modelSaveFile)
#     #Predict(configureFile, modelSaveFile, testDataFile, resultSaveFile)







############################################################# only find clusters
def getClu(dismat, K = 200):
    '''
    只寻找聚类中心，最后在聚类
    :param dismat:
    :param K:
    :return:
    '''
    proNum = len(dismat)
    centers = get_max_index(dismat)
    for i in range(3, K+1):
        print("find the center:", i)
        disArray = [0.0]*proNum
        for u in range(proNum):
            if u not in centers:
                dis = 100000
                for v in range(len(centers)):
                    tmp = getdis(u, centers[v], dismat)
                    if dis > tmp:
                        dis = tmp
                disArray[u] = dis
        # print("disArray.index(max(disArray)) = ", disArray)
        centers.append(disArray.index(max(disArray)))
    print("all centers :", centers)
    res = [[] for i in range(K)]
    print("Start clustering...")
    for i in range(proNum):
        disCenter = [0.0]*len(centers)
        for j in range(len(centers)):
            disCenter[j] = getdis(i, centers[j], dismat)
        res[disCenter.index(min(disCenter))].append(i)

    return res



#####################################################对用户聚类相关函数

def get_userdismat(upmat):
    "输入为用户-节目矩阵，输出为节目距离矩阵"
    print('start get_userdismat...')
    n = len(upmat[0])
    user_num = len(upmat)
    dismat = [[0]*user_num for i in range(user_num)]
    mx = np.asarray(upmat)
    for i in range(user_num):
        print('the ',i,' times...')
        for j in range(i+1,user_num):
            # dismat[i][j] = cal_dist(col_upmat, i, j)
            dismat[i][j] = sum((abs(mx[i]-mx[j])))
    return dismat



def getuserClu(dismat, K = 200):
    '''
    只寻找聚类中心，最后在聚类
    :param dismat:
    :param K:
    :return:
    '''
    proNum = len(dismat)
    centers = get_max_index(dismat)
    for i in range(3, K+1):
        print("find the center:", i)
        disArray = [0.0]*proNum
        for u in range(proNum):
            if u not in centers:
                dis = 10000000
                for v in range(len(centers)):
                    tmp = getdis(u, centers[v], dismat)
                    if dis > tmp:
                        dis = tmp
                disArray[u] = dis
        # print("disArray.index(max(disArray)) = ", disArray)
        centers.append(disArray.index(max(disArray)))
    print("all centers :", centers)
    res = [[] for i in range(K)]
    print("Start clustering...")
    for i in range(proNum):
        disCenter = [0.0]*len(centers)
        for j in range(len(centers)):
            disCenter[j] = getdis(i, centers[j], dismat)
        res[disCenter.index(min(disCenter))].append(i)

    return res


def get_newuser_pcmat(result, user_num):
    '''

    :param result: 聚类结果
    :param user_num: 数据中节目的个数
    :return:3000*200矩阵，表明节目归属哪一个聚类
    '''
    print('start get_new_pcmat...')
    clu_num = len(result)
    new_pcmat = [[0]*clu_num for i in range(user_num)]
    for i in range(user_num):
        for j in range(clu_num):
            if i in result[j]:
                new_pcmat[i][j] = 1
                break
    return new_pcmat


def usertopn(upmat_train, MAXC, clu_num, new_pcmat):
    "根据聚类结果得到用户-节目类别矩阵, 用户数为4000， 聚类数为MAXC, 通过new_upmat对每一行降序排列，就可以得到用户喜欢的topN节目,最后的new_upmat保存的信息是每个用户对每一类节目的喜欢度，就是把一个类别的节目的观看时长比加和在了一起"
    print('start topn...')
    user_num = len(upmat_train)
    pro_num = len(upmat_train[0])
    l2 = len(clu_num)
    ind = 0
    new_upmat = []
    # for i in range(user_num):
    #     print('#######################################',i)
    #     print('l1')
    #     s = [0]*MAXC
    #     for j in range(pro_num):
    #         tmp = [x*upmat_train[i][j] for x in new_pcmat[j]]
    #         s = [s[i]+tmp[i]  for i in range(MAXC)]
    #     res = 0
    #     print('l2')
    #     for i in range(MAXC):
    #         res += pow(s[i],2)
    #     print('l3')
    #     fm = math.sqrt(res)
    #     if fm != 0:
    #         s = [x/fm for x in s]
    #     print('l4')
    #     new_upmat.append(s)
    for j in range(pro_num):
        print('#######################################',j)
        print('l1')
        s = [0]*MAXC
        s = np.asarray(s)
        new_pcmat = np.asarray(new_pcmat)
        for i in range(user_num):
            tmp = upmat_train[i][j]*new_pcmat[j]
            s = s+tmp
        print('l2')
        res = sum(pow(s,2))
        print('l3')
        fm = math.sqrt(res)
        if fm != 0:
            s = s/fm
        print('l4')
        s = s.tolist()
        new_upmat.append(s)

    return new_upmat



############################################################# only find clusters
def getMulClu(dismat, MAXC , clu_num):
    pro_num = len(dismat)
    data = [i for i in range(pro_num)]
    centers = get_max_index(dismat)
    test = get_max_index_in(data, dismat)
    print('test my method-----------------------', centers, '  and  ', test)
    nodes = centers
    clu_res = []
    #第一组，选定最远的两个点最为初始簇中心
    print('centers[0]:', centers)
    class_num = 2
    while True:
        print('class_num:', class_num)
        clu_res.append([nodes[0]])
        clu_res.append([nodes[1]])
        l = len(data)
        Dis = [0.0]*2
        for i in range(l):
            if data[i] in nodes:
                continue
            for j in range(2):
                Dis[j] = getdis(data[i], nodes[j], dismat)
            if Dis[0] > Dis[1]:
                clu_res[class_num-1].append(data[i])
            else:
                clu_res[class_num-2].append(data[i])

        if class_num == MAXC:
            break

        max_clu_num = -1
        max_clu_ind = -1
        for i in range(class_num):
            if len(clu_res[i]) > max_clu_num:
                max_clu_num = len(clu_res[i])
                max_clu_ind = i
        data = clu_res[max_clu_ind]
        del clu_res[max_clu_ind]
        del centers[max_clu_ind]
        nodes = get_max_index_in(data, dismat)
        centers += nodes

        class_num += 1

    minData = []
    class_num = 0
    while class_num < len(clu_res):
        if len(clu_res[class_num]) <= clu_num:
            minData += clu_res[class_num]
            del centers[class_num]
            del clu_res[class_num]
        else :
            class_num += 1

    print("len(centers):", len(centers))
    pcmat = [[0.0]*len(centers) for i in range(pro_num)]
    # for i in range(pro_num):
    #     for j in range(len(centers)):
    #         pcmat[i][j] = getdis(i, centers[j], dismat)

    cnt = [0]*len(centers)
    for i in range(pro_num):
        tmp = [1000000.0]*len(centers)
        for j in range(len(centers)):
            tmp[j] = getdis(i, centers[j], dismat)
        index = 0
        while index < 1:
            minIndex = tmp.index(min(tmp))
            # pcmat[i][minIndex] = min(tmp)
            cnt[minIndex] =  cnt[minIndex] + 1
            pcmat[i][minIndex] = 1
            index = index + 1
            tmp[minIndex] = 1000000.0

    # pcmat = np.asarray(pcmat)
    # pcmat = 1/pcmat
    # pcmat = pcmat.tolist()
    f = open('D:\毕设\接收\data\cntMul.txt','w')
    for i in range(len(cnt)):
        # print('len(check_result[i]):',len(check_result[i]))
        f.write(str(cnt[i]))
        f.write('\n')
    f.close()

    # pcmat = np.asarray(pcmat)
    # pcmat = pcmat.transpose()
    # s = sum(pcmat)
    # pcmat = pcmat/s
    # pcmat = pcmat.transpose()
    # pcmat = 1/pcmat
    # pcmat = pcmat.tolist()
    return pcmat


def multopn(upmat_train,new_pcmat):
    "根据聚类结果得到用户-节目类别矩阵, 用户数为4000， 聚类数为MAXC, 通过new_upmat对每一行降序排列，就可以得到用户喜欢的topN节目,最后的new_upmat保存的信息是每个用户对每一类节目的喜欢度，就是把一个类别的节目的观看时长比加和在了一起"
    print('start topn...')
    user_num = len(upmat_train)
    pro_num = len(upmat_train[0])
    ind = 0
    new_upmat = []
    # for i in range(user_num):
    #     print('#######################################',i)
    #     print('l1')
    #     s = [0]*MAXC
    #     for j in range(pro_num):
    #         tmp = [x*upmat_train[i][j] for x in new_pcmat[j]]
    #         s = [s[i]+tmp[i]  for i in range(MAXC)]
    #     res = 0
    #     print('l2')
    #     for i in range(MAXC):
    #         res += pow(s[i],2)
    #     print('l3')
    #     fm = math.sqrt(res)
    #     if fm != 0:
    #         s = [x/fm for x in s]
    #     print('l4')
    #     new_upmat.append(s)
    for i in range(user_num):
        # print('#######################################',i)
        # print('l1')
        s = [0]*len(new_pcmat[0])
        s = np.asarray(s)
        new_pcmat = np.asarray(new_pcmat)
        for j in range(pro_num):
            # print(upmat_train[i][j])
            # print(type(upmat_train[i][j]))
            # print(new_pcmat[j])
            # print(type(new_pcmat[j]))
            tmp = float(upmat_train[i][j])*new_pcmat[j]
            s = s+tmp
        # print('l2')
        res = sum(pow(s,2))
        # print('l3')
        fm = math.sqrt(res)
        if fm != 0:
            s = s/fm
        # print('l4')
        s = s.tolist()
        new_upmat.append(s)

    return new_upmat


def justbuildpcmat(centers, pro_num, dismat):
    print("len(centers):", len(centers))
    pcmat = [[0.0]*len(centers) for i in range(pro_num)]
    # for i in range(pro_num):
    #     for j in range(len(centers)):
    #         pcmat[i][j] = getdis(i, centers[j], dismat)

    cnt = [0]*len(centers)
    for i in range(pro_num):
        tmp = [1000000.0]*len(centers)
        for j in range(len(centers)):
            if i == centers[j]:
                tmp[j] = 0.0000001
                continue
            tmp[j] = getdis(i, centers[j], dismat)
        index = 0
        while index < 1:
            minIndex = tmp.index(min(tmp))
            cnt[minIndex] =  cnt[minIndex] + 1
            pcmat[i][minIndex] = min(tmp)
            # pcmat[i][minIndex] = 1
            index = index + 1
            tmp[minIndex] = 1000000.0

    # pcmat = np.asarray(pcmat)
    # pcmat = 1/pcmat
    # pcmat = pcmat.tolist()
    f = open('D:\毕设\接收\data\cntMul.txt','w')
    for i in range(len(cnt)):
        # print('len(check_result[i]):',len(check_result[i]))
        f.write(str(cnt[i]))
        f.write('\n')
    f.close()

    pcmat = np.asarray(pcmat)
    pcmat = pcmat.transpose()
    s = sum(pcmat)
    pcmat = pcmat/s
    pcmat = pcmat.transpose()
    for i in range(len(pcmat)):
        for j in range(len(pcmat[0])):
            if pcmat[i][j] != 0:
                pcmat[i][j] = 1/pcmat[i][j]
    pcmat = pcmat.tolist()
    return pcmat



def userSVD(upmat, factorNum, qi, learnRate = 0.01, regularization = 0.01):
    #learnRate = 0.01, regularization = 0.01  learnRate 0.9

    #get the configure
    averageScore = Average(upmat)
    userNum = len(upmat)
    itemNum = len(upmat[0])
    bi = [0.0 for i in range(itemNum)]
    bu = [0.0 for i in range(userNum)]
    temp = math.sqrt(factorNum)
    # qi = [[(0.1 * random.random() / temp) for j in range(factorNum)] for i in range(itemNum)]
    # pu = [[(0.1 * random.random() / temp)  for j in range(factorNum)] for i in range(userNum)]
    # qi = [[random.uniform(-1,1) for j in range(factorNum)] for i in range(itemNum)]
    pu = [[random.uniform(-1,1)   for j in range(factorNum)] for i in range(userNum)]
    qi = np.array(qi)
    pu = np.array(pu)
    upmat = np.array(upmat)
    print("initialization end\nstart training\n")

    outMatrix = [[0.0 for j in range(itemNum)] for i in range(userNum)]
    print("userNum = ", userNum, " ,iterNum = ", itemNum)
    #train model
    preRmse = 1000000.0
    for step in range(100):
        for uid in range(userNum):
            # print("uid = ", uid)
            for iid in range(itemNum):
                score = upmat[uid][iid]
                prediction = averageScore + bu[uid] + bi[iid] + np.dot(pu[uid],qi[iid])
                # prediction = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
                eui = score - prediction
                #update parameters
                bu[uid] += learnRate * (eui - regularization * bu[uid])
                bi[iid] += learnRate * (eui - regularization * bi[iid])
                #print("iid = ", iid)
                tmp = copy.deepcopy(pu[uid])
                pu[uid] += learnRate*(eui*qi[iid] - regularization*pu[uid])
                # qi[iid] += learnRate*(eui*tmp - regularization*qi[iid])

                # for k in range(factorNum):
                #     temp = pu[uid][k]   #attention here, must save the value of pu before updating
                #     pu[uid][k] += learnRate * (eui * qi[iid][k] - regularization * pu[uid][k])
                #     qi[iid][k] += learnRate * (eui * temp - regularization * qi[iid][k])

        # learnRate *= 0.9
        # regularization *= 1.05

        curRmse = Validate(upmat, averageScore, bu, bi, pu, qi, userNum, itemNum)
        print("test_RMSE in step %d: %f" %(step, curRmse))
        # if curRmse >= preRmse:
        #     break
        # else:
        #     preRmse = curRmse
        preRmse = curRmse

    #write the model to files
    for uid in range(userNum):
        for iid in range(itemNum):
            outMatrix[uid][iid] = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
    print("model generation over")
    return outMatrix







###########################SVD++


def getImp(second_pcmat, ru, factorNum):
    num = math.sqrt(len(ru))
    y = [0.0]*factorNum
    y = np.asarray(y)
    for i in range(len(ru)):
        y = y + second_pcmat[i]
    y = y/num
    return y




def userSVDplusplus(upmat, factorNum, qi, learnRate = 0.003, regularization = 0.005, regularization2 = 0.015):
    #learnRate = 0.01, regularization = 0.01  learnRate 0.9

    #get the configure
    averageScore = Average(upmat)
    userNum = len(upmat)
    itemNum = len(upmat[0])
    bi = [0.0 for i in range(itemNum)]
    bu = [0.0 for i in range(userNum)]
    temp = math.sqrt(factorNum)
    # qi = [[(0.1 * random.random() / temp) for j in range(factorNum)] for i in range(itemNum)]
    # pu = [[(0.1 * random.random() / temp)  for j in range(factorNum)] for i in range(userNum)]
    # qi = [[random.uniform(-1,1) for j in range(factorNum)] for i in range(itemNum)]
    second_pcmat = [[random.uniform(0,1) for j in range(factorNum)] for i in range(itemNum)]
    pu = [[random.uniform(-1,1)   for j in range(factorNum)] for i in range(userNum)]
    second_pcmat= np.asarray(second_pcmat)
    qi = np.asarray(qi)
    pu = np.asarray(pu)
    upmat = np.asarray(upmat)


    uCnt = [[] for i in range(userNum)]
    for i in range(userNum):
        for j in range(itemNum):
            if upmat[i][j] != 0.0:
                uCnt[i].append(j)

    f = open('D:\毕设\接收\data\my_uCnt2.txt','w')
    for i in range(len(uCnt)):
        # print('len(check_result[i]):',len(check_result[i]))
        f.write(str(len(uCnt[i])))
        f.write('-------------------')
        for j in range(len(uCnt[i])):
            f.write(str(uCnt[i][j]))
            f.write(' ')
        f.write('\n')
    f.close()

    print("initialization end\nstart training\n")

    outMatrix = [[0.0 for j in range(itemNum)] for i in range(userNum)]
    print("userNum = ", userNum, " ,iterNum = ", itemNum)
    #train model
    preRmse = 1000000.0
    for step in range(30):
        for uid in range(userNum):
            # print("uid = ", uid)
            ru = len(uCnt[uid])

            for iid in range(itemNum):
                score = upmat[uid][iid]

                y = [0.0]*factorNum
                y = np.asarray(y)
                for i in range(ru):
                    y += second_pcmat[uCnt[uid][i]]
                y = y/math.sqrt(ru)

                prediction = averageScore + bu[uid] + bi[iid] + np.dot(pu[uid]+y,qi[iid])

                # prediction = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
                eui = score - prediction
                #update parameters
                bu[uid] += learnRate * (eui - regularization * bu[uid])
                bi[iid] += learnRate * (eui - regularization * bi[iid])
                tmp = copy.deepcopy(pu[uid])
                pu[uid] += learnRate*(eui*qi[iid] - regularization2*pu[uid])
                # qi[iid] += learnRate*(eui*tmp - regularization*qi[iid])
                second_pcmat[iid] += learnRate*(eui*qi[iid]/math.sqrt(ru)-regularization2*second_pcmat[uCnt[uid][random.randint(0,len(uCnt[uid])-1)]])
                # for k in range(factorNum):
                #     temp = pu[uid][k]   #attention here, must save the value of pu before updating
                #     pu[uid][k] += learnRate * (eui * qi[iid][k] - regularization * pu[uid][k])
                #     qi[iid][k] += learnRate * (eui * temp - regularization * qi[iid][k])

        learnRate *= 0.9
        # regularization *= 1.05

        curRmse = Validate(upmat, averageScore, bu, bi, pu, qi, userNum, itemNum)
        print("test_RMSE in step %d: %f" %(step, curRmse))
        # if curRmse >= preRmse:
        #     break
        # else:
        #     preRmse = curRmse
        preRmse = curRmse

    #write the model to files
    for uid in range(userNum):
        for iid in range(itemNum):
            outMatrix[uid][iid] = PredictScore(averageScore, bu[uid], bi[iid], pu[uid], qi[iid])
    print("model generation over")
    return outMatrix