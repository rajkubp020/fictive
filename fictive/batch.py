#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 18:11:02 2019

@author: rajesh
"""

import mendeleev as table
import pandas as pd
import numpy as np

def get_mass(name):
    return table.element(name).mass
    print(table.element(name).mass)

def get_things(formula):
    dict1 = {}
    for i in formula.split('-'):
        k = ''
        p = ''
        for ind,j in enumerate(i):
            try:
                float(j)
                p += j
            except:
                if j=='.':
                    p += j
                else:
                    k = i[ind:]
                    break
        dict1[k] = float(p)
    return dict1
    

def get_molecular_mass(formula):
    dict1 = get_things(formula)
    s = 0
    for key, value in dict1.items():
        s += value*molecule_mass(key)/100
    return s
    print(s)

def molecule_mass(name):
    name = name.strip()
    dict1 = {}
    k = ''
    p = '0'
    for ind,i in enumerate(name):
        if i==i.upper():
            try:
                float(i)
                p += i
                if ind==(len(name)-1):
                    dict1[k] = float(p)
            except:
                try:
                    dict1[k] = float(p)
                    p = ''
                    k = ''
                    k += i
                except:
                    dict1[k] = 1.0
                    p = ''
                    k = ''
                    k += i
        else:
            k += i

        if ind==(len(name)-1):
            try:
                float(name[-1])
            except:
                dict1[k] = float(1.0)

    dict1.pop('')

    s = 0
    for i in dict1:
        s += dict1[i]*get_mass(i)
    return s
    print(s)

def wper2molper(formula,ifdict=False):
    if ifdict:
        dict1 = formula
    else:
        dict1 = get_things(formula)
    

    dict2 = dict1.copy()
    dict3 = dict1.copy()

    for key, value in dict1.items():
        dict2[key] = value/molecule_mass(key)

    for key, value in dict2.items():
        dict3[key] = dict2[key]*100/sum(dict2.values())
    return dict3

def molper2wper(formula,ifdict=False):
    if ifdict:
        dict1 = formula
    else:
        dict1 = get_things(formula)
    dict2 = dict1.copy()
    dict3 = dict1.copy()

    for key, value in dict1.items():
        dict2[key] = dict1[key]*molecule_mass(key)
    for key, value in dict2.items():
        dict3[key] = dict2[key]*100/sum(dict2.values())

    return dict3


