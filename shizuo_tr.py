# -*- coding: utf-8 -*-

###以西瓜书76为例决策树
import pandas as pd
import numpy as np
import math
import copy

data = pd.read_excel('watermelon20.xlsx')
#data.isnull().sum()
#根节点
#y = data['好瓜'].value_counts().shape[0]
#计算信息熵
def Information_Entropy(plist):
    ent = 0
    for p in plist:
        ent += p*math.log(p,2)
    ent = -ent
    return ent

def get_p(feature,data):
    idx = data[feature].value_counts().index.tolist()
    val = data[feature].value_counts().tolist()
    sum_val = sum(val)
    plist = []
    for i in val:
        plist.append(i/sum_val)
    return idx,plist


####属性的信息增益

def Decision_Tree(data,root_node,old_features_list,entD,flag=0):
    #data:根节点未划分的数据
    #root_node:根节点
    #feature:当前属性集合（分支节点）
    #entD:根节点的信息熵
    #example for西瓜数据集2.0:
    #对于初始时，data为整个数据集，root_node为好瓜，feature为要考察的属性（分支节点）,entD为好瓜坏瓜信息熵
#        total_num = data.shape[0]##根节点D的样本个数    
    print('--------------------------------')
    print('输入根节点:',root_node)
    print('取值:',data[root_node].values[0])
    flag +=1
    if len(old_features_list) == 0:
        return 
    if entD==0:
        print('结论：坏瓜!') if data['好瓜'].values[0]=='否' else print('结论：好瓜!')
        print('--------------------------------')
        return
    gain_step1 = []
    ent_whole_step1 = []
    ent_whole_name = []
    
    #注意：如果直接将features_list输出，那么在这个分支remove了这个属性，对于其他分支也无法使用，将产生错误影响!!!!
    #因此输出和输入不是同一个列表!!!!!    
    features_list = copy.deepcopy(old_features_list)
#    features_list = old_features_list
    for feature in features_list: #对于不同属性
        grouped = data.groupby(feature)
        fe = data[feature].value_counts().index.tolist()
#        print(fe)
        subclass = []
        for f in fe:
            subclass.append(grouped.get_group(f))#按分支节点划分子类
        idx,plist = [],[]
        sub_num = []##每个子类的样本个数
        for sub in subclass:
            i,p = get_p('好瓜',sub) #计算每个子类好坏瓜的概率
            idx.append(i)
            plist.append(p)
            sub_num.append(sub.shape[0])##每个子类的样本个数
        ##多个分支节点的信息熵
        ent_whole = []  
        for per_plist in plist:
            per_ent = Information_Entropy(per_plist)
            ent_whole.append(per_ent)
            
        gain_step1.append(get_Gain(sub_num,ent_whole,entD))  #每个属性的信息增益,每个属性一个数  
        ent_whole_step1.append(ent_whole)          #这个属性几个子类的信息熵
        ent_whole_name.append(fe)                  #这个属性几个子类的信息熵对应的名字，如‘清晰’
        
        print('属性:',feature,'增益:',get_Gain(sub_num,ent_whole,entD))
   #####找到信息增益最大的，选为划分属性
    idmax = gain_step1.index(max(gain_step1))   
    root_node = features_list[idmax]  ##新的根节点“纹理”  
    entD = ent_whole_step1[idmax]
    entD_name = ent_whole_name[idmax]
    re1 = save_re(features_list,gain_step1)
    re1.to_excel(str(flag)+'.xlsx')
    print('产生新的节点:',root_node,'下一阶段可取值:',entD_name)
    print('--------------------------------')    
    features_list.remove(root_node)  ##属性中 去除新的根节点
    
    die_data = data.groupby(root_node)
    
#    new_depart = data[root_node].value_counts().index.tolist()
#    for i in range(len(new_depart)):
#        print('根节点:',root_node,'------------','取值:',new_depart)
#        Decision_Tree(die_data.get_group(new_depart[i]),root_node,features_list,entD[i],flag)

    return die_data,root_node,features_list,entD,entD_name
        
#        ent_whole_2 = ent_whole_step1[idmax]
        
def get_Gain(sub_num,ent_whole,entD):##信息增益
    add = 0
    total_num = sum(sub_num)
    for i in range(len(ent_whole)):
        add += (sub_num[i]/total_num)*ent_whole[i]
    return entD - add
#        Gain = get_Gain(sub_num,ent_whole,entD)
        
##保存每个属性的增益结果
def save_re(features_list,gain_step):
    features_listcp,gain_stepcp = features_list.copy(),gain_step.copy()
    features_listcp = np.array(features_listcp)
    gain_stepcp = np.array(gain_step)
    
    re = np.hstack((features_listcp.reshape((features_listcp.shape[0],1)),gain_stepcp.reshape((features_listcp.shape[0],1))))
    return pd.DataFrame(re,columns=['属性','信息增益'])

        
##以好瓜为根节点，计算各个属性的信息增益
features_list = ['色泽','根蒂','敲声','纹理','脐部','触感']
root_node = '好瓜'

idx,plist = get_p('好瓜',data)
entD = Information_Entropy(plist)

##############
print('第0层')
##根节点为好瓜
die_data,root_node,fl,entD1,entDname1 = Decision_Tree(data,root_node,features_list,entD)

##############
#print('第1层')
##根节点为纹理
node1 = Decision_Tree(die_data.get_group(entDname1[0]),root_node,fl,entD1[0])
node2 = Decision_Tree(die_data.get_group(entDname1[1]),root_node,fl,entD1[1])
node3 = Decision_Tree(die_data.get_group(entDname1[2]),root_node,fl,entD1[2])
#
###############
print('第2层')
#根节点为根蒂
node11 = Decision_Tree(node1[0].get_group(node1[4][0]),node1[1],node1[2],node1[3][0])
node12 = Decision_Tree(node1[0].get_group(node1[4][1]),node1[1],node1[2],node1[3][1])
node13 = Decision_Tree(node1[0].get_group(node1[4][2]),node1[1],node1[2],node1[3][2])
#
#根节点为触感
node21 = Decision_Tree(node2[0].get_group(node2[4][0]),node2[1],node2[2],node2[3][0])
node22 = Decision_Tree(node2[0].get_group(node2[4][1]),node2[1],node2[2],node2[3][1])
#
###############
print('第3层')
node31 = Decision_Tree(node12[0].get_group(node12[4][0]),node12[1],node12[2],node12[3][0])
node32 = Decision_Tree(node12[0].get_group(node12[4][1]),node12[1],node12[2],node12[3][1])
#
###############
print('第4层')
node41 = Decision_Tree(node31[0].get_group(node31[4][0]),node31[1],node31[2],node31[3][0])
node41 = Decision_Tree(node31[0].get_group(node31[4][1]),node31[1],node31[2],node31[3][1])
#


