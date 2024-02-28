
import torch
import numpy as np
from time import time

import random as rd
import multiprocessing
import heapq
import metrics

from functools import partial


def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.AUC(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)#100
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)#从item_score中返回前k大的itemscore对应的item id

    r = []
    for i in K_max_item_score:#检查预测的前k大在不在test中
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    #auc = 0.
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def test_one_user(train_items,test_set,num,Ks,x):
    #x的结构为（rate，uid）
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items =train_items[u]#train中寻找u对应的tiems
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = test_set[u]#test中u对应的items

    all_items = set(range(num))#所有item的id

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []
    aucc=[]



    for K in Ks:#[20, 40, 60, 80, 100]不同的k对应的指标
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(metrics.hit_at_k(r, K))



    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

def test(model, users_to_test, num,args,train_items,test_set,drop_flag=False, batch_test_flag=False):
    Ks = eval(args.Ks)
    cores = multiprocessing.cpu_count() // 2
    BATCH_SIZE = args.batch_size


    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0. , 'rmse': 0. }#ks为输出的epoch

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE * 2#1024 or 512？？
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1#//为向下取整，直接返回商
    #如果有500个testuser，batchsize为512，则每个batch中有一个user

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]#分batch训练

        if batch_test_flag:
            # batch-item test
            n_item_batchs = num// i_batch_size + 1
            rate_batch = np.zeros(shape=(len(user_batch), num))

            i_count = 0
            for i_batch_id in range(n_item_batchs):
                i_start = i_batch_id * i_batch_size
                i_end = min((i_batch_id + 1) * i_batch_size, num)

                item_batch = range(i_start, i_end)

                if drop_flag == False:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=False)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
                else:
                    u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                                  item_batch,
                                                                  [],
                                                                  drop_flag=True)
                    i_rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

                rate_batch[:, i_start: i_end] = i_rate_batch
                i_count += i_rate_batch.shape[1]

            assert i_count == num

        else:
            # all-item test
            item_batch = range(num)#n_items

            if drop_flag == False:
                #user*64  item*64
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              drop_flag=False)
                #计算得分
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()
            else:
                u_g_embeddings, pos_i_g_embeddings, _ = model(user_batch,
                                                              item_batch,
                                                              [],
                                                              drop_flag=True)
                rate_batch = model.rating(u_g_embeddings, pos_i_g_embeddings).detach().cpu()

        user_batch_rating_uid = zip(rate_batch.numpy(), user_batch)#让对应列成为一组，如(rate1,uid1),(rate2,uid2)
        test_one_user_func=partial(test_one_user,train_items,test_set,num,Ks)
        batch_result = pool.map(test_one_user_func, user_batch_rating_uid)#调用进程池，执行test_one_user函数，把第二个参数的元素传入函数中执行
        count += len(batch_result)

        # err=torch.pow(rate_batch - data_generator.R[user_batch,:],2)
        # mse=torch.sum(err)/u_batch_size
        # rmse1=torch.sqrt(mse)
        # print('rmse',rmse1)
        # print('rmse',metrics.mse(rate_batch,data_generator.R[user_batch,:]))
        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users

        print('test_:', result)

    assert count == n_test_users
    pool.close()
    return result