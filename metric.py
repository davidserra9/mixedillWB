# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

'''
import matplotlib
#matplotlib.use('agg')
import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from skimage.metrics import structural_similarity as ssim
from skimage import color
import logging
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
import lpips
np.set_printoptions(threshold=sys.maxsize)

class deltaE00():
    def __init__(self, color_chart_area=0):
        super().__init__()
        self.color_chart_area = color_chart_area
        self.kl = 1
        self.kc = 1
        self.kh = 1

    def compute(self, img1, img2):
        """ Compute the deltaE00 between two numpy RGB images
        From M. Afifi: https://github.com/mahmoudnafifi/WB_sRGB/blob/master/WB_sRGB_Python/evaluation/calc_deltaE2000.py
        :param img1: numpy RGB image
        :param img2: numpy RGB image
        :return: deltaE00
        """

        # Convert to Lab
        img1 = color.rgb2lab(img1)
        img2 = color.rgb2lab(img2)

        # reshape to 1D array
        img1 = img1.reshape(-1, 3).astype(np.float32)
        img2 = img2.reshape(-1, 3).astype(np.float32)

        # compute deltaE00
        Lstd = np.transpose(img1[:, 0])
        astd = np.transpose(img1[:, 1])
        bstd = np.transpose(img1[:, 2])
        Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
        Lsample = np.transpose(img2[:, 0])
        asample = np.transpose(img2[:, 1])
        bsample = np.transpose(img2[:, 2])
        Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
        Cabarithmean = (Cabstd + Cabsample) / 2
        G = 0.5 * (1 - np.sqrt((np.power(Cabarithmean, 7)) / (np.power(
            Cabarithmean, 7) + np.power(25, 7))))
        apstd = (1 + G) * astd
        apsample = (1 + G) * asample
        Cpsample = np.sqrt(np.power(apsample, 2) + np.power(bsample, 2))
        Cpstd = np.sqrt(np.power(apstd, 2) + np.power(bstd, 2))
        Cpprod = (Cpsample * Cpstd)
        zcidx = np.argwhere(Cpprod == 0)
        hpstd = np.arctan2(bstd, apstd)
        hpstd[np.argwhere((np.abs(apstd) + np.abs(bstd)) == 0)] = 0
        hpsample = np.arctan2(bsample, apsample)
        hpsample = hpsample + 2 * np.pi * (hpsample < 0)
        hpsample[np.argwhere((np.abs(apsample) + np.abs(bsample)) == 0)] = 0
        dL = (Lsample - Lstd)
        dC = (Cpsample - Cpstd)
        dhp = (hpsample - hpstd)
        dhp = dhp - 2 * np.pi * (dhp > np.pi)
        dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
        dhp[zcidx] = 0
        dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
        Lp = (Lsample + Lstd) / 2
        Cp = (Cpstd + Cpsample) / 2
        hp = (hpstd + hpsample) / 2
        hp = hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi
        hp = hp + (hp < 0) * 2 * np.pi
        hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
        Lpm502 = np.power((Lp - 50), 2)
        Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
        Sc = 1 + 0.045 * Cp
        T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + \
            0.32 * np.cos(3 * hp + np.pi / 30) \
            - 0.20 * np.cos(4 * hp - 63 * np.pi / 180)
        Sh = 1 + 0.015 * Cp * T
        delthetarad = (30 * np.pi / 180) * np.exp(
            - np.power((180 / np.pi * hp - 275) / 25, 2))
        Rc = 2 * np.sqrt((np.power(Cp, 7)) / (np.power(Cp, 7) + np.power(25, 7)))
        RT = - np.sin(2 * delthetarad) * Rc
        klSl = self.kl * Sl
        kcSc = self.kc * Sc
        khSh = self.kh * Sh
        de00 = np.sqrt(np.power((dL / klSl), 2) + np.power((dC / kcSc), 2) +
                       np.power((dH / khSh), 2) + RT * (dC / kcSc) * (dH / khSh))

        return np.sum(de00) / (np.shape(de00)[0] - self.color_chart_area)


def calc_mae(source, target, color_chart_area=0):
  source = np.reshape(source, [-1, 3]).astype(np.float32)
  target = np.reshape(target, [-1, 3]).astype(np.float32)
  source_norm = np.sqrt(np.sum(np.power(source, 2), 1))
  target_norm = np.sqrt(np.sum(np.power(target, 2), 1))
  norm = source_norm * target_norm
  L = np.shape(norm)[0]
  inds = norm != 0
  angles = np.sum(source[inds, :] * target[inds, :], 1) / norm[inds]
  angles[angles > 1] = 1
  f = np.arccos(angles)
  f[np.isnan(f)] = 0
  f = f * 180 / np.pi
  return sum(f)

def trimean(values):
    sorted_values = sorted(values)
    n = len(sorted_values)
    median = sorted_values[n // 2]

    if n % 2 == 0:
        lower_half = sorted_values[:n // 2]
        upper_half = sorted_values[n // 2:]
        q1 = (lower_half[len(lower_half) // 2 - 1] + lower_half[len(lower_half) // 2]) / 2
        q3 = (upper_half[len(upper_half) // 2 - 1] + upper_half[len(upper_half) // 2]) / 2
    else:
        lower_half = sorted_values[:n // 2]
        upper_half = sorted_values[n // 2 + 1:]
        q1 = lower_half[len(lower_half) // 2]
        q3 = upper_half[len(upper_half) // 2]

    trimean = (q1 + 2 * median + q3) / 4
    return trimean
