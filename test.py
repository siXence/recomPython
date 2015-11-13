from __future__ import print_function
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import clusters
import math
import re
import random

#coding=utf-8
#__author__ = 'xinxvwin'



# def cluster_process(dismat, upmat, max_clu, clu_min_num):
#     centers = get_max_index(dismat)
#     class_num = 2
#     Clusters =



# filepath = 'D:\毕设\接收\data\综合采样2.txt'
# # original_data = clusters.get_orginal_sample(filepath)
# [upmat,original_data,proid2seq, upmat_test] = clusters.pre_process(filepath)
#
# # dismat = clusters.get_dismat(upmat)
# # f = open('D:\毕设\接收\data\my_dismat2.txt','w')
# # for i in range(len(dismat)):
# #     # print('len(check_result[i]):',len(check_result[i]))
# #     for j in range(len(dismat[0])):
# #         f.write(str(dismat[i][j]))
# #         f.write(' ')
# #     f.write('\n')
# # f.close()
# f = open('D:\毕设\接收\data\my_dismat2.txt','r')
# alllines = f.readlines()
# f.close()
# dismat = []
# for eachline in alllines:
#     dismat.append(eachline.split(' '))
# for i in range(len(dismat)):
#     for j in range(len(dismat[0])-1):
#         dismat[i][j] = float(dismat[i][j])
# for i in range(len(dismat)):
#     dismat[i] = dismat[i][:-1]
#
#
# result = clusters.cluster_process(300, dismat, upmat, 10)
# # result = clusters.my_maxminclustart(dismat, 50)
# f = open('D:\毕设\接收\data\clu_result2.txt','w')
# l1 = len(result)
# for i in range(l1):
#     for j in range(len(result[i])):
#         f.write(str(result[i][j]))
#         f.write('   ')
#     f.write('\n')
# f.close()
# check_result = [[] for i in range(300)]
# print('len_check_result:',len(check_result))
# all_pro = []
# pro_id = []
# f = open('D:\毕设\接收\data\综合节目2.txt','r')
# alllines = f.readlines()
# f.close()
# for eachline in alllines:
#     all_pro.append(eachline)
#     pro_id.append(int((eachline.split('\t')[0])))
# print('len_all_pro:',len(all_pro))
# print('len_all_pro:',all_pro[0])
# l1 = len(pro_id)
# l2 = len(result)
# print('l1:',l1)
# print('l2:',l2)
# for i in range(l1):
#     # print('check_result:',check_result[0])
#     for j in range(l2):
#         if proid2seq[pro_id[i]] in result[j]:
#             check_result[j].append(all_pro[i])
#             break
#             # print(check_result[j][-1])
# print('check_result:',len(check_result[0]))
# print('check_result:',len(check_result[1]))
# f = open('D:\毕设\接收\data\check_result2.txt','w')
# l1 = len(check_result)
# print('l1:',l1)
# for i in range(l1):
#     f.write('第')
#     f.write(str(i+1))
#     f.write('类：**************************************************************************')
#     f.write('\n')
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(check_result[i])):
#         f.write(check_result[i][j])
#         f.write('\n')
# f.close()
# new_pcmat = clusters.get_new_pcmat(result, len(upmat[0]))
# new_upmat = clusters.topn(upmat, 300, result, new_pcmat)
# # de_order_clusters = clusters.get_topn(new_upmat)
# # print('start get_topn...8')
# # [users_precision, users_recall, allusers_precision, allusers_recall] = clusters.cal_precision_recall(de_order_clusters, result, upmat_test, upmat)
# # x = [i for i in range(1, 201)]
# # y = allusers_precision[0:200]
# # z = allusers_recall[0:200]
# # print(allusers_recall[50])
# # pl.plot(x,y,'ro--')
# # pl.plot(x,z,'go--')
# # pl.show()
# # print('finised')
# f = open('D:\毕设\接收\data\my_upmat2.txt','w')
# for i in range(len(upmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat[0])):
#         f.write(str(upmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# trainset = clusters.get_set(upmat)
#
# f = open('D:\毕设\接收\data\my_trainset2.txt','w')
# for i in range(len(trainset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(trainset[0])):
#         f.write(str(trainset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
# f = open('D:\毕设\接收\data\my_upmat_test2.txt','w')
# for i in range(len(upmat_test)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat_test[0])):
#         f.write(str(upmat_test[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# testset = clusters.get_set(upmat_test)
#
# f = open('D:\毕设\接收\data\my_testset2.txt','w')
# for i in range(len(testset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(testset[0])):
#         f.write(str(testset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# f = open('D:\毕设\接收\data\my_new_pcmat2.txt','w')
# for i in range(len(new_pcmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(new_pcmat[0])):
#         f.write(str(new_pcmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
#
# UPrec = clusters.get_UPrec(new_upmat,new_pcmat)
#
# f = open('D:\毕设\接收\data\my_UPrec2.txt','w')
# for i in range(len(UPrec)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(UPrec[0])):
#         f.write(str(UPrec[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# [PRE, RECALL] = clusters.uptest(UPrec, trainset, testset)
# x = [i for i in range(1, 101)]
# y = PRE[0:100]
# z = RECALL[0:100]
# pl.plot(x,y,'ro--')
# pl.plot(x,z,'go--')
# pl.show()
# print('finised')



# a = np.random.random([2,3])
#
# print(a)

# #FCM聚类方法
#
# filepath = 'D:\毕设\接收\data\综合采样2.txt'
# # original_data = clusters.get_orginal_sample(filepath)
# [upmat,original_data,proid2seq, upmat_test] = clusters.pre_process(filepath)
# [Cluster_Res, result] = clusters.fcm_clusting(upmat, 200, 2, pow(math.e, -40))
# f = open('D:\毕设\接收\data\clu_result2_fcm.txt','w')
# l1 = len(result)
# for i in range(l1):
#     for j in range(len(result[i])):
#         f.write(str(result[i][j]))
#         f.write('   ')
#     f.write('\n')
# f.close()
# f = open('D:\毕设\接收\data\Cluster_Res2_fcm.txt','w')
# mytmp = sum(np.asarray(Cluster_Res)).tolist()
# for i in range(len(mytmp)):
#     f.write(str(mytmp[i]))
#     f.write('   ')
# f.close()
# check_result = [[] for i in range(200)]
# print('len_check_result:',len(check_result))
# all_pro = []
# pro_id = []
# f = open('D:\毕设\接收\data\综合节目2.txt','r')
# alllines = f.readlines()
# f.close()
# for eachline in alllines:
#     all_pro.append(eachline)
#     pro_id.append(int((eachline.split('\t')[0])))
# print('len_all_pro:',len(all_pro))
# print('len_all_pro:',all_pro[0])
# l1 = len(pro_id)
# l2 = len(result)
# print('l1:',l1)
# print('l2:',l2)
# for i in range(l1):
#     # print('check_result:',check_result[0])
#     for j in range(l2):
#         if proid2seq[pro_id[i]] in result[j]:
#             check_result[j].append(all_pro[i])
#             break
#             # print(check_result[j][-1])
# print('check_result:',len(check_result[0]))
# print('check_result:',len(check_result[1]))
# f = open('D:\毕设\接收\data\check_result2_fcm.txt','w')
# l1 = len(check_result)
# print('l1:',l1)
# for i in range(l1):
#     f.write('第')
#     f.write(str(i+1))
#     f.write('类：**************************************************************************')
#     f.write('\n')
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(check_result[i])):
#         f.write(check_result[i][j])
#         f.write('\n')
# f.close()
# new_pcmat = clusters.get_new_pcmat(result, len(upmat[0]))
# new_upmat = clusters.topn(upmat, 200, result, new_pcmat)
# # de_order_clusters = clusters.get_topn(new_upmat)
# # print('start get_topn...8')
# # [users_precision, users_recall, allusers_precision, allusers_recall] = clusters.cal_precision_recall(de_order_clusters, result, upmat_test, upmat)
# # x = [i for i in range(1, 201)]
# # y = allusers_precision[0:200]
# # z = allusers_recall[0:200]
# # print(allusers_recall[50])
# # pl.plot(x,y,'ro--')
# # pl.plot(x,z,'go--')
# # pl.show()
# # print('finised')
# f = open('D:\毕设\接收\data\my_upmat2_fcm.txt','w')
# for i in range(len(upmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat[0])):
#         f.write(str(upmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# trainset = clusters.get_set(upmat)
#
# f = open('D:\毕设\接收\data\my_trainset2_fcm.txt','w')
# for i in range(len(trainset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(trainset[0])):
#         f.write(str(trainset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
# f = open('D:\毕设\接收\data\my_upmat_test2_fcm.txt','w')
# for i in range(len(upmat_test)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat_test[0])):
#         f.write(str(upmat_test[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# testset = clusters.get_set(upmat_test)
#
# f = open('D:\毕设\接收\data\my_testset2_fcm.txt','w')
# for i in range(len(testset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(testset[0])):
#         f.write(str(testset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# f = open('D:\毕设\接收\data\my_new_pcmat2_fcm.txt','w')
# for i in range(len(new_pcmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(new_pcmat[0])):
#         f.write(str(new_pcmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
#
# UPrec = clusters.get_UPrec(new_upmat,new_pcmat)
#
# f = open('D:\毕设\接收\data\my_UPrec2_fcm.txt','w')
# for i in range(len(UPrec)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(UPrec[0])):
#         f.write(str(UPrec[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# [PRE, RECALL] = clusters.uptest(UPrec, trainset, testset)
# x = [i for i in range(1, 101)]
# y = PRE[0:100]
# z = RECALL[0:100]
# pl.plot(x,y,'ro--')
# pl.plot(x,z,'go--')
# pl.show()
# print('finised')


#
# f = open('D:\毕设\接收\data\my_dismat2.txt','r')
# alllines = f.readlines()
# f.close()
# dismat = []
# for eachline in alllines:
#     dismat.append(eachline.split(' '))
# for i in range(len(dismat)):
#     for j in range(len(dismat[0])-1):
#         dismat[i][j] = float(dismat[i][j])
# for i in range(len(dismat)):
#     dismat[i] = dismat[i][:-1]




# ##################################################################多中心
# filepath = 'D:\毕设\接收\data\综合采样2.txt'
# # original_data = clusters.get_orginal_sample(filepath)
# [upmat,original_data,proid2seq, upmat_test] = clusters.pre_process(filepath)
#
# # dismat = clusters.get_cos_dismat(upmat)
# # f = open('D:\毕设\接收\data\my_dismat2_cos.txt','w')
# # for i in range(len(dismat)):
# #     # print('len(check_result[i]):',len(check_result[i]))
# #     for j in range(len(dismat[0])):
# #         f.write(str(dismat[i][j]))
# #         f.write(' ')
# #     f.write('\n')
# # f.close()
# f = open('D:\毕设\接收\data\my_dismat2.txt','r')
# alllines = f.readlines()
# f.close()
# dismat = []
# for eachline in alllines:
#     dismat.append(eachline.split(' '))
# for i in range(len(dismat)):
#     for j in range(len(dismat[0])-1):
#         dismat[i][j] = float(dismat[i][j])
# for i in range(len(dismat)):
#     dismat[i] = dismat[i][:-1]
#
#
# # result = clusters.cluster_process(300, dismat, upmat, 10)
# result = clusters. multi_centers_max_min_cluster(dismat)
# # result = clusters.my_maxminclustart(dismat, 50)
# f = open('D:\毕设\接收\data\clu_result2.txt','w')
# l1 = len(result)
# for i in range(l1):
#     for j in range(len(result[i])):
#         f.write(str(result[i][j]))
#         f.write('   ')
#     f.write('\n')
# f.close()
# check_result = [[] for i in range(len(result))]
# print('len_check_result:',len(check_result))
# all_pro = []
# pro_id = []
# f = open('D:\毕设\接收\data\综合节目2.txt','r')
# alllines = f.readlines()
# f.close()
# for eachline in alllines:
#     all_pro.append(eachline)
#     pro_id.append(int((eachline.split('\t')[0])))
# print('len_all_pro:',len(all_pro))
# print('len_all_pro:',all_pro[0])
# l1 = len(pro_id)
# l2 = len(result)
# print('l1:',l1)
# print('l2:',l2)
# for i in range(l1):
#     # print('check_result:',check_result[0])
#     for j in range(l2):
#         if proid2seq[pro_id[i]] in result[j]:
#             check_result[j].append(all_pro[i])
#             break
#             # print(check_result[j][-1])
# print('check_result:',len(check_result[0]))
# print('check_result:',len(check_result[1]))
# f = open('D:\毕设\接收\data\check_result2.txt','w')
# l1 = len(check_result)
# print('l1:',l1)
# for i in range(l1):
#     f.write('第')
#     f.write(str(i+1))
#     f.write('类：**************************************************************************')
#     f.write('\n')
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(check_result[i])):
#         f.write(check_result[i][j])
#         f.write('\n')
# f.close()
# new_pcmat = clusters.get_new_pcmat(result, len(upmat[0]))
# new_upmat = clusters.topn(upmat, len(result), result, new_pcmat)
# # de_order_clusters = clusters.get_topn(new_upmat)
# # print('start get_topn...8')
# # [users_precision, users_recall, allusers_precision, allusers_recall] = clusters.cal_precision_recall(de_order_clusters, result, upmat_test, upmat)
# # x = [i for i in range(1, 201)]
# # y = allusers_precision[0:200]
# # z = allusers_recall[0:200]
# # print(allusers_recall[50])
# # pl.plot(x,y,'ro--')
# # pl.plot(x,z,'go--')
# # pl.show()
# # print('finised')
# f = open('D:\毕设\接收\data\my_upmat2.txt','w')
# for i in range(len(upmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat[0])):
#         f.write(str(upmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# trainset = clusters.get_set(upmat)
#
# f = open('D:\毕设\接收\data\my_trainset2.txt','w')
# for i in range(len(trainset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(trainset[0])):
#         f.write(str(trainset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
# f = open('D:\毕设\接收\data\my_upmat_test2.txt','w')
# for i in range(len(upmat_test)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat_test[0])):
#         f.write(str(upmat_test[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# testset = clusters.get_set(upmat_test)
#
# f = open('D:\毕设\接收\data\my_testset2.txt','w')
# for i in range(len(testset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(testset[0])):
#         f.write(str(testset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# f = open('D:\毕设\接收\data\my_new_pcmat2.txt','w')
# for i in range(len(new_pcmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(new_pcmat[0])):
#         f.write(str(new_pcmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
#
# UPrec = clusters.get_UPrec(new_upmat,new_pcmat)
#
# f = open('D:\毕设\接收\data\my_UPrec2.txt','w')
# for i in range(len(UPrec)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(UPrec[0])):
#         f.write(str(UPrec[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# [PRE, RECALL] = clusters.uptest(UPrec, trainset, testset)
# x = [i for i in range(1, 101)]
# y = PRE[0:100]
# z = RECALL[0:100]
# pl.plot(x,y,'ro--')
# pl.plot(x,z,'go--')
# pl.show()
# print('finised')




# ##################################################################最终采用方案
# filepath = 'D:\毕设\接收\data\综合采样2.txt'
# # original_data = clusters.get_orginal_sample(filepath)
# [upmat,original_data,proid2seq, upmat_test] = clusters.pre_process(filepath)
#
# dismat = clusters.get_dismat(upmat)
# f = open('D:\毕设\接收\data\my_dismat2.txt','w')
# for i in range(len(dismat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(dismat[0])):
#         f.write(str(dismat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
# # f = open('D:\毕设\接收\data\my_dismat1.txt','r')
# # alllines = f.readlines()
# # f.close()
# # dismat = []
# # for eachline in alllines:
# #     dismat.append(eachline.split(' '))
# # for i in range(len(dismat)):
# #     for j in range(len(dismat[0])-1):
# #         dismat[i][j] = float(dismat[i][j])
# # for i in range(len(dismat)):
# #     dismat[i] = dismat[i][:-1]
#
#
# # [result, centers] = clusters.cluster_process(300, dismat, upmat, 10)
# # result = clusters.adjust_centers_clusters(dismat, upmat, 200)
# # result = clusters.desprade_cluster(dismat, 300)
# # result = clusters.desprade_cluster2(dismat, 300)
# # result = clusters.desprade_cluster3(dismat, 300)
# result = clusters.desprade_cluster_min(dismat, 400, 4)
# # [result, centers] = clusters. multi_centers_max_min_cluster(dismat)
# # result = clusters.my_maxminclustart(dismat, 50)
# f = open('D:\毕设\接收\data\clu_result2.txt','w')
# l1 = len(result)
# for i in range(l1):
#     for j in range(len(result[i])):
#         f.write(str(result[i][j]))
#         f.write('   ')
#     f.write('\n')
# f.close()
# check_result = [[] for i in range(len(result))]
# print('len_check_result:',len(check_result))
# all_pro = []
# pro_id = []
# f = open('D:\毕设\接收\data\综合节目2.txt','r')
# alllines = f.readlines()
# f.close()
# for eachline in alllines:
#     all_pro.append(eachline)
#     pro_id.append(int((eachline.split('\t')[0])))
# print('len_all_pro:',len(all_pro))
# print('len_all_pro:',all_pro[0])
# l1 = len(pro_id)
# l2 = len(result)
# print('l1:',l1)
# print('l2:',l2)
# for i in range(l1):
#     # print('check_result:',check_result[0])
#     for j in range(l2):
#         if proid2seq[pro_id[i]] in result[j]:
#             check_result[j].append(all_pro[i])
#             break
#             # print(check_result[j][-1])
# print('check_result:',len(check_result[0]))
# print('check_result:',len(check_result[1]))
# f = open('D:\毕设\接收\data\check_result2.txt','w')
# l1 = len(check_result)
# print('l1:',l1)
# for i in range(l1):
#     f.write('第')
#     f.write(str(i+1))
#     f.write('类：**************************************************************************')
#     f.write('\n')
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(check_result[i])):
#         f.write(check_result[i][j])
#         f.write('\n')
# f.close()
# new_pcmat = clusters.get_new_pcmat(result, len(upmat[0]))
# new_upmat = clusters.topn(upmat, len(result), result, new_pcmat)
# # new_pcmat = clusters.get_pcmat(centers, len(upmat[0]), dismat)
# # new_upmat = clusters.get_ucmat(upmat, new_pcmat)
#
# f = open('D:\毕设\接收\data\my_upmat2.txt','w')
# for i in range(len(upmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat[0])):
#         f.write(str(upmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# trainset = clusters.get_set(upmat)
#
# f = open('D:\毕设\接收\data\my_trainset2.txt','w')
# for i in range(len(trainset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(trainset[0])):
#         f.write(str(trainset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
# f = open('D:\毕设\接收\data\my_upmat_test2.txt','w')
# for i in range(len(upmat_test)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat_test[0])):
#         f.write(str(upmat_test[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# testset = clusters.get_set(upmat_test)
#
# f = open('D:\毕设\接收\data\my_testset2.txt','w')
# for i in range(len(testset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(testset[0])):
#         f.write(str(testset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# f = open('D:\毕设\接收\data\my_new_pcmat2.txt','w')
# for i in range(len(new_pcmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(new_pcmat[0])):
#         f.write(str(new_pcmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
#
# UPrec = clusters.get_UPrec(new_upmat,new_pcmat)
#
# f = open('D:\毕设\接收\data\my_UPrec2.txt','w')
# for i in range(len(UPrec)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(UPrec[0])):
#         f.write(str(UPrec[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# [PRE, RECALL] = clusters.uptest(UPrec, trainset, testset)
# x = [i for i in range(1, 101)]
# y = PRE[0:100]
# z = RECALL[0:100]
# pl.plot(x,y,'ro--')
# pl.plot(x,z,'go--')
# pl.show()
# print('finised')











# ##################################### SVD ############################################################
#
# f = open('D:\毕设\接收\data\my_upmat1.txt','r')
# alllines = f.readlines()
# f.close()
# upmat = []
# for eachline in alllines:
#     upmat.append(eachline.split(' '))
# for i in range(len(upmat)):
#     for j in range(len(upmat[0])-1):
#         upmat[i][j] = float(upmat[i][j])
# for i in range(len(upmat)):
#     upmat[i] = upmat[i][:-1]
#
# UPrec = clusters.SVD(upmat, 223)
#
# f = open('D:\毕设\接收\data\my_UPrec1.txt','w')
# for i in range(len(UPrec)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(UPrec[0])):
#         f.write(str(UPrec[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# print('finish the SVD')





# ##################################################################最终采用方案?????
# filepath = 'D:\毕设\接收\data\综合采样1.txt'
# # original_data = clusters.get_orginal_sample(filepath)
# [upmat,original_data,proid2seq, upmat_test] = clusters.pre_process(filepath)
#
# # dismat = clusters.get_dismat(upmat)
# # f = open('D:\毕设\接收\data\my_dismat1.txt','w')
# # for i in range(len(dismat)):
# #     # print('len(check_result[i]):',len(check_result[i]))
# #     for j in range(len(dismat[0])):
# #         f.write(str(dismat[i][j]))
# #         f.write(' ')
# #     f.write('\n')
# # f.close()
# f = open('D:\毕设\接收\data\my_dismat1.txt','r')
# alllines = f.readlines()
# f.close()
# dismat = []
# for eachline in alllines:
#     dismat.append(eachline.split(' '))
# for i in range(len(dismat)):
#     for j in range(len(dismat[0])-1):
#         dismat[i][j] = float(dismat[i][j])
# for i in range(len(dismat)):
#     dismat[i] = dismat[i][:-1]
#
#
#
# result = clusters.getClu(dismat)
#
# print("result : ", result)
#
#
# f = open('D:\毕设\接收\data\clu_result1.txt','w')
# l1 = len(result)
# for i in range(l1):
#     for j in range(len(result[i])):
#         f.write(str(result[i][j]))
#         f.write('   ')
#     f.write('\n')
# f.close()
# check_result = [[] for i in range(len(result))]
# print('len_check_result:',len(check_result))
# all_pro = []
# pro_id = []
# f = open('D:\毕设\接收\data\综合节目1.txt','r')
# alllines = f.readlines()
# f.close()
# for eachline in alllines:
#     all_pro.append(eachline)
#     pro_id.append(int((eachline.split('\t')[0])))
# print('len_all_pro:',len(all_pro))
# print('len_all_pro:',all_pro[0])
# l1 = len(pro_id)
# l2 = len(result)
# print('l1:',l1)
# print('l2:',l2)
# for i in range(l1):
#     # print('check_result:',check_result[0])
#     for j in range(l2):
#         if proid2seq[pro_id[i]] in result[j]:
#             check_result[j].append(all_pro[i])
#             break
#             # print(check_result[j][-1])
# print('check_result:',len(check_result[0]))
# print('check_result:',len(check_result[1]))
# f = open('D:\毕设\接收\data\check_result1.txt','w')
# l1 = len(check_result)
# print('l1:',l1)
# for i in range(l1):
#     f.write('第')
#     f.write(str(i+1))
#     f.write('类：**************************************************************************')
#     f.write('\n')
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(check_result[i])):
#         f.write(check_result[i][j])
#         f.write('\n')
# f.close()
# new_pcmat = clusters.get_new_pcmat(result, len(upmat[0]))
# new_upmat = clusters.topn(upmat, len(result), result, new_pcmat)
# # new_pcmat = clusters.get_pcmat(centers, len(upmat[0]), dismat)
# # new_upmat = clusters.get_ucmat(upmat, new_pcmat)
#
# f = open('D:\毕设\接收\data\my_upmat1.txt','w')
# for i in range(len(upmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat[0])):
#         f.write(str(upmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# trainset = clusters.get_set(upmat)
#
# f = open('D:\毕设\接收\data\my_trainset1.txt','w')
# for i in range(len(trainset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(trainset[0])):
#         f.write(str(trainset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
# f = open('D:\毕设\接收\data\my_upmat_test1.txt','w')
# for i in range(len(upmat_test)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat_test[0])):
#         f.write(str(upmat_test[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# testset = clusters.get_set(upmat_test)
#
# f = open('D:\毕设\接收\data\my_testset1.txt','w')
# for i in range(len(testset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(testset[0])):
#         f.write(str(testset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# f = open('D:\毕设\接收\data\my_new_pcmat1.txt','w')
# for i in range(len(new_pcmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(new_pcmat[0])):
#         f.write(str(new_pcmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
#
# UPrec = clusters.get_UPrec(new_upmat,new_pcmat)
#
# f = open('D:\毕设\接收\data\my_UPrec1.txt','w')
# for i in range(len(UPrec)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(UPrec[0])):
#         f.write(str(UPrec[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# [PRE, RECALL] = clusters.uptest(UPrec, trainset, testset)
# x = [i for i in range(1, 101)]
# y = PRE[0:100]
# z = RECALL[0:100]
# pl.plot(x,y,'ro--')
# pl.plot(x,z,'go--')
# pl.show()
# print('finised')






# ##################################################################对人聚类
# filepath = 'D:\毕设\接收\data\综合采样1.txt'
# # original_data = clusters.get_orginal_sample(filepath)
# [upmat,original_data,proid2seq, upmat_test] = clusters.pre_process(filepath)
#
# dismat = clusters.get_userdismat(upmat)
# f = open('D:\毕设\接收\data\my_userdismat1.txt','w')
# for i in range(len(dismat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(dismat[0])):
#         f.write(str(dismat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
# # f = open('D:\毕设\接收\data\my_userdismat1.txt','r')
# # alllines = f.readlines()
# # f.close()
# # dismat = []
# # for eachline in alllines:
# #     dismat.append(eachline.split(' '))
# # for i in range(len(dismat)):
# #     for j in range(len(dismat[0])-1):
# #         dismat[i][j] = float(dismat[i][j])
# # for i in range(len(dismat)):
# #     dismat[i] = dismat[i][:-1]
#
#
#
# result = clusters.getClu(dismat)
#
# print("result : ", result)
#
#
# f = open('D:\毕设\接收\data\clu_user_result1.txt','w')
# l1 = len(result)
# for i in range(l1):
#     for j in range(len(result[i])):
#         f.write(str(result[i][j]))
#         f.write('   ')
#     f.write('\n')
# f.close()
# check_result = [[] for i in range(len(result))]
# print('len_check_result:',len(check_result))
# all_pro = []
# pro_id = []
# f = open('D:\毕设\接收\data\综合节目1.txt','r')
# alllines = f.readlines()
# f.close()
# for eachline in alllines:
#     all_pro.append(eachline)
#     pro_id.append(int((eachline.split('\t')[0])))
# print('len_all_pro:',len(all_pro))
# print('len_all_pro:',all_pro[0])
# l1 = len(pro_id)
# l2 = len(result)
# print('l1:',l1)
# print('l2:',l2)
# for i in range(l1):
#     # print('check_result:',check_result[0])
#     for j in range(l2):
#         if proid2seq[pro_id[i]] in result[j]:
#             check_result[j].append(all_pro[i])
#             break
#             # print(check_result[j][-1])
# print('check_result:',len(check_result[0]))
# print('check_result:',len(check_result[1]))
# f = open('D:\毕设\接收\data\check_user_result1.txt','w')
# l1 = len(check_result)
# print('l1:',l1)
# for i in range(l1):
#     f.write('第')
#     f.write(str(i+1))
#     f.write('类：**************************************************************************')
#     f.write('\n')
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(check_result[i])):
#         f.write(check_result[i][j])
#         f.write('\n')
# f.close()
# new_upmat = clusters.get_newuser_pcmat(result, len(upmat[0]))
# new_pcmat = clusters.usertopn(upmat, len(result), result, new_upmat)
# # new_pcmat = clusters.get_pcmat(centers, len(upmat[0]), dismat)
# # new_upmat = clusters.get_ucmat(upmat, new_pcmat)
#
# f = open('D:\毕设\接收\data\my_userupmat1.txt','w')
# for i in range(len(upmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat[0])):
#         f.write(str(upmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# trainset = clusters.get_set(upmat)
#
# f = open('D:\毕设\接收\data\myuser_trainset1.txt','w')
# for i in range(len(trainset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(trainset[0])):
#         f.write(str(trainset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
# f = open('D:\毕设\接收\data\my_upmat_test1.txt','w')
# for i in range(len(upmat_test)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(upmat_test[0])):
#         f.write(str(upmat_test[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# testset = clusters.get_set(upmat_test)
#
# f = open('D:\毕设\接收\data\myuser_testset1.txt','w')
# for i in range(len(testset)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(testset[0])):
#         f.write(str(testset[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# f = open('D:\毕设\接收\data\myuser_new_pcmat1.txt','w')
# for i in range(len(new_pcmat)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(new_pcmat[0])):
#         f.write(str(new_pcmat[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
#
# UPrec = clusters.get_UPrec(new_upmat,new_pcmat)
#
# f = open('D:\毕设\接收\data\myuser_UPrec1.txt','w')
# for i in range(len(UPrec)):
#     # print('len(check_result[i]):',len(check_result[i]))
#     for j in range(len(UPrec[0])):
#         f.write(str(UPrec[i][j]))
#         f.write(' ')
#     f.write('\n')
# f.close()
#
# [PRE, RECALL] = clusters.uptest(UPrec, trainset, testset)
# x = [i for i in range(1, 101)]
# y = PRE[0:100]
# z = RECALL[0:100]
# pl.plot(x,y,'ro--')
# pl.plot(x,z,'go--')
# pl.show()
# print('finised')




##################################################################
f = open('D:\毕设\接收\data\my_upmat2.txt','r')
alllines = f.readlines()
f.close()
upmat = []
for eachline in alllines:
    upmat.append(eachline.split(' '))
for i in range(len(upmat)):
    for j in range(len(upmat[0])-1):
        upmat[i][j] = float(upmat[i][j])
for i in range(len(upmat)):
    upmat[i] = upmat[i][:-1]


f = open('D:\毕设\接收\data\my_dismat2.txt','r')
alllines = f.readlines()
f.close()
dismat = []
for eachline in alllines:
    dismat.append(eachline.split(' '))
for i in range(len(dismat)):
    for j in range(len(dismat[0])-1):
        dismat[i][j] = float(dismat[i][j])
for i in range(len(dismat)):
    dismat[i] = dismat[i][:-1]

#their method
# [result, centers] = clusters.cluster_process(300, dismat, upmat, 10)
# # new_pcmat = clusters.get_new_pcmat(result, len(upmat[0]))
# new_pcmat = clusters.justbuildpcmat(centers, len(upmat[0]), dismat)
# new_upmat = clusters.topn(upmat, len(result), result, new_pcmat)

# #my method
# result = clusters.desprade_cluster_min(dismat, 400, 4)
# new_pcmat = clusters.get_new_pcmat(result, len(upmat[0]))
# new_upmat = clusters.topn(upmat, len(result), result, new_pcmat)
# print("len(result):", len(result));




## 多维
# new_pcmat = clusters.getMulClu(dismat, 400, 4)
# new_upmat = clusters.multopn(upmat,new_pcmat)


# new_upmat = clusters.topn(upmat, 222, [], new_pcmat)

#SVD
[result, centers] = clusters.cluster_process(200, dismat, upmat, 10)
# new_pcmat = clusters.get_new_pcmat(result, len(upmat[0]))
new_pcmat = clusters.justbuildpcmat(centers, len(upmat[0]), dismat)
UPrec = clusters.userSVDplusplus(upmat, len(result), new_pcmat)


# UPrec = clusters.get_UPrec(new_upmat,new_pcmat)

f = open('D:\毕设\接收\data\my_UPrec2.txt','w')
for i in range(len(UPrec)):
    # print('len(check_result[i]):',len(check_result[i]))
    for j in range(len(UPrec[0])):
        f.write(str(UPrec[i][j]))
        f.write(' ')
    f.write('\n')
f.close()

# [PRE, RECALL] = clusters.uptest(UPrec, trainset, testset)
# x = [i for i in range(1, 101)]
# y = PRE[0:100]
# z = RECALL[0:100]
# pl.plot(x,y,'ro--')
# pl.plot(x,z,'go--')
# pl.show()
print('finised')
