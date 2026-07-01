import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

nodes = []
initializers = []

# ------------------------------------------------------------
# Utility helpers 
# ------------------------------------------------------------

def add_initializer(name, array):
    initializers.append(
        numpy_helper.from_array(array.astype(np.float32), name)
    )

def gather_channel(inp, channel, out):
    idx_name = f"idx_{channel}"

    if idx_name not in {i.name for i in initializers}:
        initializers.append(
            numpy_helper.from_array(
                np.array([channel], dtype=np.int64),
                idx_name
            )
        )

    nodes.append(
        helper.make_node(
            "Gather",
            [inp, idx_name],
            [out],
            axis=1
        )
    )


def mul(a, b, out):
    nodes.append(
        helper.make_node(
            "Mul",
            [a, b],
            [out]
        )
    )

border = np.zeros((1,1,30,30), dtype=np.float32)

border[:,:,0,:] = 1
border[:,:,-1,:] = 1
border[:,:,:,0] = 1
border[:,:,:,-1] = 1

add_initializer("border_const", border)
    
def conv(inp, weight_name, out, pads=(1,1,1,1)):
    nodes.append(
        helper.make_node(
            "Conv",
            inputs=[inp, weight_name],
            outputs=[out],
            pads=list(pads),
            strides=[1,1]
        )
    )

kernel = np.array(
    [[[
        [0,1,0],
        [1,0,1],
        [0,1,0]
    ]]],
    dtype=np.float32
)

add_initializer("cross_kernel", kernel)

def add(a, b, out):
    nodes.append(
        helper.make_node(
            "Add",
            [a, b],
            [out]
        )
    )

add_initializer(
    "zero_scalar",
    np.array(0, dtype=np.float32)
)

def greater(a, b, out):
    nodes.append(
        helper.make_node(
            "Greater",
            [a, b],
            [out]
        )
    )

def cast(inp, out):
    nodes.append(
        helper.make_node(
            "Cast",
            [inp],
            [out],
            to=TensorProto.FLOAT
        )
    )

def propagate(current, iteration):
    neighbor = f"neighbor_sum_{iteration}"
    expanded = f"expanded_sum_{iteration}"
    expanded_bool = f"expanded_bool_{iteration}"
    expanded_mask = f"expanded_mask_{iteration}"
    reach = f"reachable_{iteration}"
    conv(current, "cross_kernel", neighbor)
    add(neighbor, current, expanded)
    greater(expanded, "zero_scalar", expanded_bool)
    cast(expanded_bool, expanded_mask)
    mul(expanded_mask, "background_mask", reach)
    return reach

def sub(a, b, out):
    nodes.append(
        helper.make_node(
            "Sub",
            [a, b],
            [out]
        )
    )

def concat(inputs, out):
    nodes.append(
        helper.make_node(
            "Concat",
            inputs,
            [out],
            axis=1
        )
    )
# ------------------------------------------------------------
# Stage 1 
# ------------------------------------------------------------

gather_channel("input", 0, "background_mask")

gather_channel("input", 3, "color3_mask")

mul(
    "background_mask",
    "border_const",
    "border_seed"
)

reach = "border_seed"

for i in range(60):
    reach = propagate(reach, i)


sub(
    "background_mask",
    reach,
    "holes"
)

sub(
    "background_mask",
    "holes",
    "background_new"
)

gather_channel("input", 4, "color4_mask")
add(
    "color4_mask",
    "holes",
    "color4_new"
)

gather_channel("input", 1, "channel1")
gather_channel("input", 2, "channel2")
gather_channel("input", 5, "channel5")
gather_channel("input", 6, "channel6")
gather_channel("input", 7, "channel7")
gather_channel("input", 8, "channel8")
gather_channel("input", 9, "channel9")

concat(
    [
        "background_new",
        "channel1",
        "channel2",
        "color3_mask",
        "color4_new",
        "channel5",
        "channel6",
        "channel7",
        "channel8",
        "channel9"
    ],
    "output"
)

outputs = [
    helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1,10,30,30]
    )
]
input_info = helper.make_tensor_value_info(
    "input",
    TensorProto.FLOAT,
    [1,10,30,30]
)

graph = helper.make_graph(
    nodes,
    "task002",
    [input_info],
    outputs,
    initializer=initializers
)

def make_model(graph):
    model = helper.make_model(
        graph,
        producer_name="Neurogolf",
        opset_imports=[helper.make_opsetid("", 21)]
    )

    # Use an IR version compatible with ONNX Runtime
    model.ir_version = 10
    return model

model = make_model(graph)

onnx.checker.check_model(model)
onnx.save(model, "task002.onnx")
