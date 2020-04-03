# Copyright 2020 Oath Inc. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.
import onnx
import numpy as np
from onnx import helper, TensorProto

output_type = helper.make_tensor_value_info('y', TensorProto.FLOAT, [])

node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['y'],
    value=onnx.helper.make_tensor(
        name='const_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=(),
        vals=[0.42]
    ),
)
graph_def = onnx.helper.make_graph(
    nodes = [node],
    name = 'constant_test',
    inputs = [],
    outputs = [output_type]
)
model_def = helper.make_model(graph_def, producer_name='const.py')
onnx.save(model_def, 'const.onnx')
