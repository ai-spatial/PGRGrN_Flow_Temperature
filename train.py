# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:38:50 2019

@author: xiaoweijia
"""

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import random
from model import MyModel
# run only using CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

''' Declare constants '''
learning_rate = 0.002 #0.005
learning_rate_pre = 0.005
epochs = 100
epochs_pre = 200 #200#100#200#70
batch_size = 2000
state_size = 20 
input_size = 20-9
# phy_size = 2
phy_size = 6#+1+2 # new
T = 13149
cv_idx = 2 # 0,1,2 for cross validation, 4383 lenth for each one
npic = 14
n_steps = int(4900/npic) # cut it to 16 pieces #43 #12 #46 
n_classes = 1 
N_sec = (npic-1)*2+1
N_seg = 456
N_res = 16
kb=1.0
keep_prob = 1.0
#s_perc = 0.001

n_upd = 5
n_fupd = 5
upd_lr = 0.005



'''initialize variables'''
tf.reset_default_graph()
random.seed(9001)
# Graph input/output
x_tr = tf.placeholder("float", [None, n_steps, input_size])  # tf.float32
y1_tr = tf.placeholder("float", [None, n_steps])  # tf.int32
m1_tr = tf.placeholder("float", [None, n_steps])

A = tf.placeholder("float", [None, None])
Ar = tf.placeholder("float", [None, None])

# new upstream
Au = tf.placeholder("float", [None, None])

# A weight 1 or 0
Ap = tf.placeholder("float", [None, state_size])

# p1 = tf.placeholder("float", [None,n_steps,phy_size-3])
p1 = tf.placeholder("float", [None, phy_size])
p2 = tf.placeholder("float", [None, n_steps, 3])
# pp = tf.concat([p1,p2],axis=-1)


Wg2 = tf.get_variable('W_g2', [state_size, state_size], tf.float32,
                      tf.random_normal_initializer(stddev=0.02))
bg2 = tf.get_variable('b_g2', [state_size], tf.float32,
                      initializer=tf.constant_initializer(0.0))

###new
Wg3 = tf.get_variable('W_g3', [state_size, state_size], tf.float32,
                      tf.random_normal_initializer(stddev=0.02))

###upstream
Wg4 = tf.get_variable('W_g4', [state_size, state_size], tf.float32,
                      tf.random_normal_initializer(stddev=0.02))
###upstream 20*9
Wg5 = tf.get_variable('W_g5', [state_size, phy_size], tf.float32,
                      tf.random_normal_initializer(stddev=0.02))
###new p 9*9
Wg6 = tf.get_variable('W_g6', [3, state_size], tf.float32,
                      tf.random_normal_initializer(stddev=0.02))

Wg7 = tf.get_variable('W_g7', [state_size, state_size], tf.float32,
                      tf.random_normal_initializer(stddev=0.02))

###new
bg3 = tf.get_variable('b_g3', [state_size], tf.float32,
                      initializer=tf.constant_initializer(0.0))

###upstream
bg4 = tf.get_variable('b_g4', [state_size], tf.float32,
                      initializer=tf.constant_initializer(0.0))

bg5 = tf.get_variable('b_g5', [state_size], tf.float32,
                      initializer=tf.constant_initializer(0.0))

Wi = tf.get_variable('Wi', [state_size, state_size], tf.float32,
                     tf.random_normal_initializer(stddev=0.02))

Wf = tf.get_variable('Wf', [state_size, state_size], tf.float32,
                     tf.random_normal_initializer(stddev=0.02))

Wo = tf.get_variable('Wo', [state_size, state_size], tf.float32,
                     tf.random_normal_initializer(stddev=0.02))

Wg = tf.get_variable('Wg', [state_size, state_size], tf.float32,
                     tf.random_normal_initializer(stddev=0.02))

Ws = tf.get_variable('Ws', [state_size, state_size], tf.float32,
                     tf.random_normal_initializer(stddev=0.02))
###new
Ws1 = tf.get_variable('Ws1', [state_size, state_size], tf.float32,
                      tf.random_normal_initializer(stddev=0.02))

Ui = tf.get_variable('Ui', [input_size, state_size], tf.float32,
                     tf.random_normal_initializer(stddev=0.02))

Uf = tf.get_variable('Uf', [input_size, state_size], tf.float32,
                     tf.random_normal_initializer(stddev=0.02))

Uo = tf.get_variable('Uo', [input_size, state_size], tf.float32,
                     tf.random_normal_initializer(stddev=0.02))

Ug = tf.get_variable('Ug', [input_size, state_size], tf.float32,
                     tf.random_normal_initializer(stddev=0.02))

Us = tf.get_variable('Us', [input_size, state_size], tf.float32,
                     tf.random_normal_initializer(stddev=0.02))
###new
Us1 = tf.get_variable('Us1', [input_size, state_size], tf.float32,
                      tf.random_normal_initializer(stddev=0.02))

###new
# Up = tf.get_variable('Up',[phy_size, state_size], tf.float32,
#                                  tf.random_normal_initializer(stddev=0.02))

w_fin = tf.get_variable('w_fin', [state_size, n_classes], tf.float32,
                        tf.random_normal_initializer(stddev=0.02))
b_fin = tf.get_variable('b_fin', [n_classes], tf.float32,
                        initializer=tf.constant_initializer(0.0))

w_f1 = tf.get_variable('w_f1', [phy_size, 1], tf.float32,
                       tf.random_normal_initializer(stddev=0.02))
b_f1 = tf.get_variable('b_f1', [1], tf.float32,
                       initializer=tf.constant_initializer(0.0))

w_f2 = tf.get_variable('w_f2', [phy_size, 1], tf.float32,
                       tf.random_normal_initializer(stddev=0.02))
b_f2 = tf.get_variable('b_f2', [1], tf.float32,
                       initializer=tf.constant_initializer(0.0))



''' Load Model and Loss Function '''

my_model = MyModel()
h_tr,pred_tr = my_model.forward(x_tr, p1, p2, Ui, Uf, Us, Wi,
                               Wf, Ws, w_f1, b_f1, w_f2, b_f2,
                               Uo, Ug, Wg6, bg5,Ar,Wg7,bg3,Us1,
                               Ws1,Wo,Wg,A,Wg2,bg2,Wg4,Au,bg4,
                               Wg3,Ap,w_fin,b_fin)

cost = my_model.loss_measure(pred_tr,y1_tr,m1_tr)

saver = tf.train.Saver(max_to_keep=3)


tvars = tf.trainable_variables()
for i in tvars:
    print(i)

gr = tf.gradients(cost, tvars)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.apply_gradients(zip(gr, tvars))

''' Load data '''
feat = np.load('./processed_features.npy')
obs = np.load('./obs_temp.npy')  # np.load('obs_temp.npy')
# sim = np.load('../pretrainer/sim_temp_composite.npy')
sim = obs
maso = (obs != -11).astype(int)
mask = (sim != -11).astype(int)

feat = np.delete(feat, [1, 2, 6, 7, 9, 10, 12, 14, 15], 2)

adj_up = np.load('./up_full.npy')
adj_dn = np.load('./dn_full.npy')

adj = adj_up  # +adj_dn
mean_adj = np.mean(adj[adj != 0])
std_adj = np.std(adj[adj != 0])
adj[adj != 0] = adj[adj != 0] - mean_adj
adj[adj != 0] = adj[adj != 0] / std_adj
adj[adj != 0] = 1 / (1 + np.exp(adj[adj != 0]))

A_hat = adj.copy()
A_hat[A_hat == np.nan] = 0
D = np.sum(A_hat, axis=1)
D[D == 0] = 1
D_inv = D ** -1.0
D_inv = np.diag(D_inv)
A_hat = np.matmul(D_inv, A_hat)
# Calculate the degree matrix and its inverse matrix, and normalize A_hat.
I = np.eye(adj.shape[0])
A_hat = A_hat + I

res_feat = np.load('./res_feat.npy')
res_rel = np.load('./res_rel.npy')

phy = np.zeros([N_res, res_rel.shape[1], phy_size + 3])
for i in range(res_feat.shape[1]):
    res_feat[:, i] /= np.max(res_feat[:, i])
for i in range(phy.shape[1]):
    phy[:, i, :-3] = res_feat  # tile operation

r = res_rel
r[r == -11] = 0
r = np.log(1 + r)
phy[:, :, -3:] = r

# up
adj_res = np.load('./up_res_full.npy')
def_val = 999999
mean_adj = np.mean(adj_res[adj_res != def_val])
std_adj = np.std(adj_res[adj_res != def_val])
adj_res[adj_res != def_val] = adj_res[adj_res != def_val] - mean_adj
adj_res[adj_res != def_val] = adj_res[adj_res != def_val] / std_adj
adj_res[adj_res != def_val] = 1 / (1 + np.exp(adj_res[adj_res != def_val]))

A_res = adj_res.copy()
A_res[A_res == def_val] = 0

# down
adj_up = np.load('./dn_res_full.npy')
# adj_res1[adj_res1==999999]=0
def_val = 999999
mean_up = np.mean(adj_up[adj_up != def_val])
std_up = np.std(adj_up[adj_up != def_val])
adj_up[adj_up != def_val] = adj_up[adj_up != def_val] - mean_up
adj_up[adj_up != def_val] = adj_up[adj_up != def_val] / std_up
adj_up[adj_up != def_val] = 1 / (1 + np.exp(adj_up[adj_up != def_val]))

A_up = adj_up.copy()
A_up[A_up == def_val] = 0

A_up = np.transpose(A_up)

# AP
# Ap0 = np.zeros((N_seg,state_size))
Ap1 = np.ones((N_seg, state_size))

x_te = feat[:, cv_idx * 4900:(cv_idx + 1) * 4900, :]
o_te = obs[:, cv_idx * 4900:(cv_idx + 1) * 4900]
mo_te = maso[:, cv_idx * 4900:(cv_idx + 1) * 4900]
y_te = sim[:, cv_idx * 4900:(cv_idx + 1) * 4900]
m_te = mask[:, cv_idx * 4900:(cv_idx + 1) * 4900]
p_te = phy[:, cv_idx * 4900:(cv_idx + 1) * 4900, :]

# np.save('./results/obs_full_upstream.npy',o_te)

if cv_idx == 1:
    x_tr_1 = feat[:, :4900, :]
    o_tr_1 = obs[:, :4900]
    mo_tr_1 = maso[:, :4900]
    y_tr_1 = sim[:, :4900]
    m_tr_1 = mask[:, :4900]
    p_tr_1 = phy[:, :4900, :]

    x_tr_2 = feat[:, 2 * 4900:3 * 4900, :]
    o_tr_2 = obs[:, 2 * 4900:3 * 4900]
    mo_tr_2 = maso[:, 2 * 4900:3 * 4900]
    y_tr_2 = sim[:, 2 * 4900:3 * 4900]
    m_tr_2 = mask[:, 2 * 4900:3 * 4900]
    p_tr_2 = phy[:, 2 * 4900:3 * 4900, :]

if cv_idx == 2:
    x_tr_1 = feat[:, :4900, :]
    o_tr_1 = obs[:, :4900]
    mo_tr_1 = maso[:, :4900]
    y_tr_1 = sim[:, :4900]
    m_tr_1 = mask[:, :4900]
    p_tr_1 = phy[:, :4900, :]

    x_tr_2 = feat[:, 4900:2 * 4900, :]
    o_tr_2 = obs[:, 4900:2 * 4900]
    mo_tr_2 = maso[:, 4900:2 * 4900]
    y_tr_2 = sim[:, 4900:2 * 4900]
    m_tr_2 = mask[:, 4900:2 * 4900]
    p_tr_2 = phy[:, 4900:2 * 4900, :]

x_train_1 = np.zeros([N_seg * N_sec, n_steps, input_size])
o_train_1 = np.zeros([N_seg * N_sec, n_steps])
mo_train_1 = np.zeros([N_seg * N_sec, n_steps])
y_train_1 = np.zeros([N_seg * N_sec, n_steps])
m_train_1 = np.zeros([N_seg * N_sec, n_steps])
p_train_1 = np.zeros([N_res * N_sec, n_steps, phy_size + 3])

x_train_2 = np.zeros([N_seg * N_sec, n_steps, input_size])
o_train_2 = np.zeros([N_seg * N_sec, n_steps])
mo_train_2 = np.zeros([N_seg * N_sec, n_steps])
y_train_2 = np.zeros([N_seg * N_sec, n_steps])
m_train_2 = np.zeros([N_seg * N_sec, n_steps])
p_train_2 = np.zeros([N_res * N_sec, n_steps, phy_size + 3])

x_test = np.zeros([N_seg * N_sec, n_steps, input_size])
o_test = np.zeros([N_seg * N_sec, n_steps])
mo_test = np.zeros([N_seg * N_sec, n_steps])
y_test = np.zeros([N_seg * N_sec, n_steps])
m_test = np.zeros([N_seg * N_sec, n_steps])
p_test = np.zeros([N_res * N_sec, n_steps, phy_size + 3])

for i in range(1, N_sec + 1):
    x_train_1[(i - 1) * N_seg:i * N_seg, :, :] = x_tr_1[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2), :]
    o_train_1[(i - 1) * N_seg:i * N_seg, :] = o_tr_1[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    mo_train_1[(i - 1) * N_seg:i * N_seg, :] = mo_tr_1[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    y_train_1[(i - 1) * N_seg:i * N_seg, :] = y_tr_1[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    m_train_1[(i - 1) * N_seg:i * N_seg, :] = m_tr_1[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    p_train_1[(i - 1) * N_res:i * N_res, :, :] = p_tr_1[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2), :]

    x_train_2[(i - 1) * N_seg:i * N_seg, :, :] = x_tr_2[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2), :]
    o_train_2[(i - 1) * N_seg:i * N_seg, :] = o_tr_2[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    mo_train_2[(i - 1) * N_seg:i * N_seg, :] = mo_tr_2[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    y_train_2[(i - 1) * N_seg:i * N_seg, :] = y_tr_2[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    m_train_2[(i - 1) * N_seg:i * N_seg, :] = m_tr_2[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    p_train_2[(i - 1) * N_res:i * N_res, :, :] = p_tr_2[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2), :]

    x_test[(i - 1) * N_seg:i * N_seg, :, :] = x_te[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2), :]
    o_test[(i - 1) * N_seg:i * N_seg, :] = o_te[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    mo_test[(i - 1) * N_seg:i * N_seg, :] = mo_te[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    y_test[(i - 1) * N_seg:i * N_seg, :] = y_te[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    m_test[(i - 1) * N_seg:i * N_seg, :] = m_te[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2)]
    p_test[(i - 1) * N_res:i * N_res, :, :] = p_te[:, int((i - 1) * n_steps / 2):int((i + 1) * n_steps / 2), :]

''' Session starts '''
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess, './models/globalmodel.ckpt')

print('Fine-tuning starts')
print('==================================')
# total_batch = int(np.floor(N_tr/batch_size))
los = 0
mre = 10
pred = np.zeros(o_test.shape)
for epoch in range(epochs):  # range(epochs):
    if np.isnan(los):
        break
    alos = 0
    alos_s = 0
    alos_p = 0

    idx = range(N_sec)
    idx = random.sample(idx, N_sec)

    for i in range(N_sec):  # better code?
        index = range(idx[i] * N_seg, (idx[i] + 1) * N_seg)
        index_p = range(idx[i] * N_res, (idx[i] + 1) * N_res)

        batch_x = x_train_1[index, :, :]
        batch_y = o_train_1[index, :]
        batch_m = mo_train_1[index, :]
        batch_p = p_train_1[index_p, :, :]

        if np.sum(batch_m) > 0:
            _, los_s = sess.run(
                [train_op, cost],
                feed_dict={
                    x_tr: batch_x,
                    y1_tr: batch_y,
                    m1_tr: batch_m,
                    A: A_hat,
                    Ar: A_res,
                    Au: A_up,
                    Ap: Ap1,
                    p1: batch_p[:, 0, :-3],
                    p2: batch_p[:, :, -3:]
                })
            alos += los
            alos_s += los_s
            if np.isnan(los):
                break
    print('Epoch ' + str(epoch) + ': loss ' + "{:.4f}".format(alos / N_sec) \
          + ': loss_s ' + "{:.4f}".format(alos_s / N_sec) \
          + ': loss_p ' + "{:.4f}".format(alos_p / N_sec))

    if np.isnan(los):
        break

    alos = 0
    alos_s = 0
    alos_p = 0
    for i in range(N_sec):  # better code?
        index = range(idx[i] * N_seg, (idx[i] + 1) * N_seg)
        index_p = range(idx[i] * N_res, (idx[i] + 1) * N_res)

        batch_x = x_train_2[index, :, :]
        batch_y = o_train_2[index, :]
        batch_m = mo_train_2[index, :]
        batch_p = p_train_2[index_p, :, :]
        #        # debug
        #        prd,cc,hh,cg,hg,oo,oh,og = sess.run(
        #            [y_prd,c,h,c_gr,h_gr,o_sr,o_hr,o_gr],
        #            feed_dict = {
        #                    x: batch_x,
        #                    y: batch_y,
        #                    m: batch_m,
        #                    keep_prob: kb,
        #                    A: A_hat
        #        })
        if np.sum(batch_m) > 0:
            _, los_s = sess.run(
                [train_op, cost],
                feed_dict={
                    x_tr: batch_x,
                    y1_tr: batch_y,
                    m1_tr: batch_m,
                    A: A_hat,
                    Ar: A_res,
                    Au: A_up,
                    Ap: Ap1,
                    p1: batch_p[:, 0, :-3],
                    p2: batch_p[:, :, -3:]

                })
            alos += los
            alos_s += los_s
            if np.isnan(los):
                break
    print('Epoch ' + str(epoch) + ': loss ' + "{:.4f}".format(alos / N_sec) \
          + ': loss_s ' + "{:.4f}".format(alos_s / N_sec) \
          + ': loss_p ' + "{:.4f}".format(alos_p / N_sec))

    # test on segments with training samples
    prd_te = np.zeros([N_sec * N_seg, n_steps, 1])

    for i in range(N_sec):  # better code?
        index = range(i * N_seg, (i + 1) * N_seg)
        index_p = range(i * N_res, (i + 1) * N_res)

        batch_x = x_test[index, :, :]
        batch_y = o_test[index, :]
        batch_m = mo_test[index, :]
        batch_p = p_test[index_p, :]

        batch_prd = sess.run(
            pred_tr,
            feed_dict={
                x_tr: batch_x,
                y1_tr: batch_y,
                m1_tr: batch_m,
                A: A_hat,
                Ar: A_res,
                Au: A_up,
                Ap: Ap1,
                p1: batch_p[:, 0, :-3],
                p2: batch_p[:, :, -3:]
            })
        prd_te[index, :, :] = batch_prd

    prd_o = np.zeros([N_seg, 4900])
    prd_o[:, :n_steps] = prd_te[0:N_seg, :, 0]

    for j in range(N_sec - 1):  # 18*125    +250 = 2500
        # st_idx = 365-(int((j+1)*365/2)-int(j*365/2))
        st_idx = n_steps - (int((j + 1) * n_steps / 2) - int(j * n_steps / 2))
        # prd_o[:, 365+int(j*365/2):365+int((j+1)*365/2)] = prd_te[(j+1)*N_seg:(j+2)*N_seg,st_idx:,0]
        prd_o[:, n_steps + int(j * n_steps / 2):n_steps + int((j + 1) * n_steps / 2)] = prd_te[
                                                                                        (j + 1) * N_seg:(j + 2) * N_seg,
                                                                                        st_idx:, 0]

    po = np.reshape(prd_o, [-1])
    ye = np.reshape(o_te, [-1])
    me = np.reshape(mo_te, [-1])
    rmse = np.sqrt(np.sum(np.square((po - ye) * me)) / np.sum(me))
    # print( 'Seg Test RMSE: '+"{:.4f}".format(rmse) )

    m_te_c = mo_te.copy()
    m_te_c[15, :] = 0
    m_te_c[50, :] = 0
    me = np.reshape(m_te_c, [-1])
    rmse_o = np.sqrt(np.sum(np.square((po - ye) * me)) / np.sum(me))
    #
    pp = np.reshape(prd_o[15, :], [-1])
    yy = np.reshape(o_te[15, :], [-1])
    mm = np.reshape(mo_te[15, :], [-1])
    rmse_15 = np.sqrt(np.sum(np.square((pp - yy) * mm)) / np.sum(mm))

    pp = np.reshape(prd_o[50, :], [-1])
    yy = np.reshape(o_te[50, :], [-1])
    mm = np.reshape(mo_te[50, :], [-1])
    rmse_50 = np.sqrt(np.sum(np.square((pp - yy) * mm)) / np.sum(mm))

    print('Test RMSE: ' + "{:.4f}".format(rmse) \
          + ', Test RMSEo: ' + "{:.4f}".format(rmse_o) \
          + ', Test RMSE15: ' + "{:.4f}".format(rmse_15) \
          + ', Test RMSE50: ' + "{:.4f}".format(rmse_50))

    if rmse < mre:
        mre = rmse
        print('saving...')
        np.save('./results/prediction_globalmodel_0130.npy', prd_o)
        saver.save(sess, './models/globalmodel_0130.ckpt')

np.save('./results/prediction_globalmodel_last_0130.npy', prd_o)

saver.save(sess, './models/globalmodel_last_0130.ckpt')

