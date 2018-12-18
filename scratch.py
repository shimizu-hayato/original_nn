#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sklearn.datasets
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np


class Function(object):
    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        return 1. - x * x

    def softmax(self, x):
        e = np.exp(x - np.max(x))  # オーバーフローを防ぐ
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:
            return e / np.array([np.sum(e, axis=1)]).T # サンプル数が1の時
        
class Newral_Network(object):
    def __init__(self, unit):
        print("Number of layer = " + str(len(unit)))
        print(unit)
        print("-------------------------")
        self.F = Function()
        self.unit = unit
        self.W = []
        self.B = []
        self.dW = []
        for i in range(len(self.unit) - 1):
            w = np.random.rand(self.unit[i], self.unit[i + 1])
            self.W.append(w)
            dw = np.random.rand(self.unit[i], self.unit[i + 1])
            self.dW.append(dw)
            b = np.random.rand(self.unit[i + 1])
            self.B.append(b)

    # 順伝搬
    def forward(self, _inputs):
        print("forward")
        self.Z = []
        self.Z.append(_inputs)
        for i in range(len(self.unit) - 1):
            u = self.U(self.Z[i], self.W[i], self.B[i])
            if(i != len(self.unit) - 2):
                z = np.tanh(u)
            else:
                z = self.F.softmax(u)
            self.Z.append(z)
        return np.argmax(z, axis=1)

    # ユニットへの総入力を返す関数
    def U(self, x, w, b):
        return np.dot(x, w) + b

    # 誤差の計算
    def calc_loss(self, label):
        error = np.sum(label * np.log(self.Z[-1]), axis=1)
        return -np.mean(error)

    # 誤差逆伝搬
    def backPropagate(self, _label, eta, M):
        # calculate output_delta and error terms
        W_grad = []
        B_grad = []
        for i in range(len(self.W)):
            w_grad = np.zeros_like(self.W[i])
            W_grad.append(w_grad)
            b_grad = np.zeros_like(self.W[i])
            B_grad.append(b_grad)

        output = True
        delta = np.zeros_like(self.Z[-1])
        for i in range(len(self.W)):
            delta = self.calc_delta(delta, self.W[-(i)], self.Z[-(i + 1)], _label, output)
            W_grad[-(i + 1)], B_grad[-(i + 1)] = self.calc_grad(self.W[-(i + 1)], self.B[-(i + 1)], self.Z[-(i + 2)], delta)

            output = False

        # パラメータのチューニング
        for i in range(len(self.W)):
            self.W[i] = self.W[i] - eta * W_grad[i] + M * self.dW[i]
            self.B[i] = self.B[i] - eta * B_grad[i]
            # モメンタムの計算
            self.dW[i] = -eta * W_grad[i] + M * self.dW[i]

    # デルタの計算
    def calc_delta(self, delta_dash, w, z, label, output):
        # delta_dash : 1つ先の層のデルタ
        # w : pre_deltaとdeltaを繋ぐネットの重み
        # z : wへ向かう出力
        print(label)
        print(z)
        if(output):
            delta = z - label
        else:
            delta = np.dot(delta_dash, w.T) * self.F.dtanh(z)
        return delta

    def train(self, dataset, N, iterations=1000, minibatch=4, eta=0.5, M=0.1):
        print("-----Train-----")
        # 入力データ[:, :self.unit[0]]
        inputs = np.array(dataset[0])
        # 訓練データ
        label = np.array(dataset[1])
        errors = []
        #print(inputs)
        #print(label)

        for val in range(iterations):
            minibatch_errors = []
            for index in range(0, N, minibatch):
                _inputs = inputs[index: index + minibatch]
                _label = label[index: index + minibatch]
                self.forward(_inputs)
                self.backPropagate(_label, eta, M)

                loss = self.calc_loss(_label)
                minibatch_errors.append(loss)
            En = sum(minibatch_errors) / len(minibatch_errors)
            print("epoch" + str(val + 1) + " : Loss = " + str(En))
            errors.append(En)
        print("\n")
        errors = np.asarray(errors)
        plt.plot(errors)


data = sklearn.datasets.make_classification(
n_features=2, n_samples=300, n_redundant=0, 
    n_informative=2,n_clusters_per_class=1, n_classes=3,random_state=0)

#plt.scatter(x[:, 0], x[:, 1],c=label,cmap=plt.cm.jet)
#plt.show()
#N=len(data[0])
#print(len(x[:, 1]))
dataset = np.c_[data[0],data[1]]
print(dataset[1][2])
Newral_Network = Newral_Network(unit=[2,3,3])
Newral_Network.train(dataset=data,N=300)