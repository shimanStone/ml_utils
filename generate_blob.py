#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/30 16:40
# @Author  : shiman
# @File    : generate_blob.py
# @describe:


model_xml = r'E:\data\model\yolo4_t_coco.xml'
# model_xml = r'E:\data\model\yolov5.xml'
model_bin = r'E:\data\model\yolo4_t_coco.bin'
# model_bin = r'E:\data\model\yolov5.bin'

import blobconverter

blob_path = blobconverter.from_openvino(
    xml=model_xml,
    bin=model_bin,
    data_type="FP16",
    shaves=6,
    version="2021.4",
    use_cache=False
)