#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 14:38
# @Author  : shiman
# @File    : test_openvino_model.py
# @describe:


from PIL import Image
import os
import sys
import numpy as np

os.environ['Path'] += 'C:\\Program Files (x86)\\Intel\\openvino_2022.1.0.643\\tools\compile_tool;'\
    'C:\Program Files (x86)\Intel\openvino_2022.1.0.643\\runtime\\bin\intel64\Release;'\
    'C:\Program Files (x86)\Intel\openvino_2022.1.0.643\\runtime\\bin\intel64\Debug;'\
    'C:\Program Files (x86)\Intel\openvino_2022.1.0.643\\runtime\\3rdparty\hddl\\bin;'\
    'C:\Program Files (x86)\Intel\openvino_2022.1.0.643\\runtime\\3rdparty\\tbb\\bin;'

from openvino.inference_engine import IECore




if __name__ == '__main__':
    # model_xml = r'E:\data\model\yolo4_t_coco.xml'
    model_xml = r'E:\data\model\yolov5.xml'
    # model_bin = r'E:\data\model\yolo4_t_coco.bin'
    model_bin = r'E:\data\model\yolov5.bin'
    test_jpg = r'E:\ml_code\pytorch_yolo4\data\street.jpg'

    if not os.path.exists(model_bin) or not os.path.exists(model_xml):
        sys.exit(-1)

    ie = IECore()
    print(f'available devices: {ie.available_devices}')

    net = ie.read_network(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.input_info))
    exec_net = ie.load_network(network=net, device_name='CPU')

    image = Image.open(test_jpg)
    image = image.resize((416,416), Image.BICUBIC)
    image = np.expand_dims(np.transpose(np.array(image, dtype='float32')/255.0, (2,0,1)), axis=0)

    output = exec_net.infer(inputs={input_blob: image})

    output = np.array(output['output'])
    print(f'output shape: {output.shape}')
    for i in range(0, output.shape[0]):
        boxes = 1


