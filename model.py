# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:38:50 2019

@author: xiaoweijia
"""

from __future__ import print_function, division
import tensorflow as tf


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

n_upd = 5
n_fupd = 5
upd_lr = 0.005


''' Build Graph '''


class MyModel(object):
    def __init__(self):
        pass



    def forward(self,x, p1, p2, Ui, Uf, Us, Wi,
                Wf, Ws, w_f1, b_f1, w_f2, b_f2,
                Uo, Ug, Wg6, bg5,Ar,Wg7,bg3,Us1,
                Ws1,Wo,Wg,A,Wg2,bg2,Wg4,Au,bg4,
                Wg3,Ap,w_fin,b_fin,reuse=False):
        f1 = tf.sigmoid(tf.matmul(p1, w_f1) + b_f1)
        f2 = tf.sigmoid(tf.matmul(p1, w_f2) + b_f2)

        filter_f1 = tf.tile(f1, [1, state_size])
        filter_f2 = tf.tile(f2, [1, state_size])

        o_sr = []

        i = tf.sigmoid(tf.matmul(x[:, 0, :], Ui))
        #  output gate
        o = tf.sigmoid(tf.matmul(x[:, 0, :], Uo))
        #  candidate cell
        g = tf.tanh(tf.matmul(x[:, 0, :], Ug))

        # new

        #    cr_pre = tf.nn.tanh( tf.matmul(cr_pre,Wg4)+ filter_f1*tf.matmul(Au,c_pre)+bg4)
        cr_pre = tf.zeros([tf.shape(p1)[0], state_size])  # tf.zeros()
        p_hid = (tf.matmul(p2[:, 0, :], Wg6) + bg5)  # N_res-h
        c_gr1 = tf.nn.tanh(tf.matmul(tf.matmul(Ar, filter_f2 * p_hid), Wg7) + bg3)  # 56*20

        #    c_gr1 = tf.nn.tanh(tf.matmul(Ar,tf.matmul(pp[:,0,:],Wg3)+bg3))
        s1 = tf.sigmoid(tf.matmul(x[:, 0, :], Us1) + tf.matmul(c_gr1, Ws1))

        c_pre = g * i + c_gr1 * s1

        h_pre = tf.tanh(c_pre) * o
        o_sr.append(h_pre)

        for t in range(1, n_steps):
            i = tf.sigmoid(tf.matmul(x[:, t, :], Ui) + tf.matmul(h_pre, Wi))
            #  forget gate
            f = tf.sigmoid(tf.matmul(x[:, t, :], Uf) + tf.matmul(h_pre, Wf))
            #  output gate
            o = tf.sigmoid(tf.matmul(x[:, t, :], Uo) + tf.matmul(h_pre, Wo))

            #  candidate cell
            g = tf.tanh(tf.matmul(x[:, t, :], Ug) + tf.matmul(h_pre, Wg))

            c_gr = tf.nn.tanh(tf.matmul(A, tf.matmul(c_pre, Wg2) + bg2))

            cr_pre = tf.nn.tanh(tf.matmul(cr_pre, Wg4) + filter_f1 * tf.matmul(Au, c_pre) + bg4)  # XXX
            p_hid = (tf.matmul(p2[:, t, :], Wg6) + tf.matmul(cr_pre, Wg3) + bg5)  # N_res-h
            c_gr1 = tf.nn.tanh(tf.matmul(tf.matmul(Ar, filter_f2 * p_hid), Wg7) + bg3)  # 56*20

            #  spatial gate
            s = tf.sigmoid(tf.matmul(x[:, t, :], Us) + tf.matmul(c_gr, Ws))
            # new
            s1 = tf.sigmoid(tf.matmul(x[:, t, :], Us1) + tf.matmul(c_gr1, Ws1))
            # new
            c_pre = c_pre * f + c_gr * s + Ap * c_gr1 * s1 + g * i
            # output state
            h_pre = tf.tanh(c_pre) * o
            o_sr.append(h_pre)

        o_sr = tf.stack(o_sr, axis=1)  # N_seg - T - state_size
        oh = tf.reshape(o_sr, [-1, state_size])

        pred = tf.matmul(oh, w_fin) + b_fin
        pred = tf.reshape(pred, [-1, n_steps, 1])

        return o_sr, pred

    def loss_measure(self,pred, y, m):
        pred_s = tf.reshape(pred, [-1, 1])
        y_s = tf.reshape(y, [-1, 1])
        m_s = tf.reshape(m, [-1, 1])
        r_cost = tf.sqrt(tf.reduce_sum(tf.square(tf.multiply((pred_s - y_s), m_s))) / tf.reduce_sum(m_s))
        return r_cost

    def apply_gr(self, Ui, Uf, Us, Wi, Wf, Ws, grads):
        Uin = Ui - upd_lr * grads[0]
        Ufn = Uf - upd_lr * grads[1]
        Usn = Us - upd_lr * grads[2]
        Win = Wi - upd_lr * grads[3]
        Wfn = Wf - upd_lr * grads[4]
        Wsn = Ws - upd_lr * grads[5]
        return Uin, Ufn, Usn, Win, Wfn, Wsn
        # updating parameters based on the grads

