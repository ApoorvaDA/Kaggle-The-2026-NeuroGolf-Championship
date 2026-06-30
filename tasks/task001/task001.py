import numpy as np
import onnx
from onnx import helper, TensorProto

#
#  Build ONNX model
#

def task001_network():

    x = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        [1,10,30,30]
    )

    y = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [1,10,30,30]
    )

    inits = []

    def const(name, arr, dtype=np.int64):
        arr = np.asarray(arr, dtype=dtype)
        inits.append(
            onnx.numpy_helper.from_array(arr, name)
        )

    #
    # Slice top-left 3x3
    #

    const("slice_starts",[0,0,0,0])
    const("slice_ends",[1,10,3,3])
    const("slice_axes",[0,1,2,3])
    const("slice_steps",[1,1,1,1])

    #
    # ReduceSum over colors 1..9
    #

    const("nz_starts",[0,1,0,0])
    const("nz_ends",[1,10,3,3])
    const("nz_axes",[0,1,2,3])
    const("nz_steps",[1,1,1,1])

    #
    # reshape pattern
    #

    const("shape_pattern",[1,10,1,3,1,3])

    #
    # reshape mask
    #

    const("shape_mask",[1,1,3,1,3,1])

    #
    # reshape after transpose
    #

    const("shape_9x9",[1,10,9,9])

    #
    # Foreground
    #
        
    const("fg_starts",[0,1,0,0])
    const("fg_ends",[1,10,9,9])
    const("fg_axes",[0,1,2,3])
    const("fg_steps",[1,1,1,1])

    const("one",
        np.array([1],dtype=np.float32),
        np.float32)


    nodes = [

        #
        # patch = input[:,:,:3,:3]
        #

        helper.make_node(
            "Slice",
            [
                "input",
                "slice_starts",
                "slice_ends",
                "slice_axes",
                "slice_steps"
            ],
            ["patch"]
        ),

        #
        # colors 1..9
        #

        helper.make_node(
            "Slice",
            [
                "patch",
                "nz_starts",
                "nz_ends",
                "nz_axes",
                "nz_steps"
            ],
            ["nonzero_channels"]
        ),

        #
        # occupancy mask
        #

        helper.make_node(
            "ReduceSum",
            ["nonzero_channels"],
            ["mask_sum"],
            axes=[1],
            keepdims=1
        ),

        #
        # mask > 0
        #

        helper.make_node(
           "Sign",
            ["mask_sum"],
            ["mask"]
        ),

        #
        # pattern
        # [1,10,3,3]
        # ->
        # [1,10,1,3,1,3]
        #

        helper.make_node(
            "Reshape",
            ["patch","shape_pattern"],
            ["pattern6"]
        ),

        #
        # mask
        # [1,1,3,3]
        # ->
        # [1,1,3,1,3,1]
        #

        helper.make_node(
            "Reshape",
            ["mask","shape_mask"],
            ["mask6"]
        ),

        #
        # broadcast multiply
        #

        helper.make_node(
            "Mul",
            ["pattern6","mask6"],
            ["kron6"]
        ),

        #
        # reorder dimensions
        #
        # [B,C,R,I,S,J]
        #
        # ->
        #
        # [B,C,R,S,I,J]
        #

        helper.make_node(
            "Transpose",
            ["kron6"],
            ["kron6_t"],
            perm=[0,1,2,3,4,5]
        ),

        #
        # collapse
        #
        # [1,10,3,3,3,3]
        #
        # ->
        #
        # [1,10,9,9]
        #

        helper.make_node(
            "Reshape",
            ["kron6_t","shape_9x9"],
            ["small_out"]
        ),

       
        helper.make_node(
            "Slice",
            [
                "small_out",
                "fg_starts",
                "fg_ends",
                "fg_axes",
                "fg_steps"
            ],
            ["foreground"]
        ),

        helper.make_node(
            "ReduceSum",
            ["foreground"],
            ["fg_sum"],
            axes=[1],
            keepdims=1
        ),

        helper.make_node(
            "Sub",
            ["one","fg_sum"],
            ["background"]
        ),

        helper.make_node(
            "Concat",
            ["background","foreground"],
            ["small_out_fixed"],
            axis=1
        ),

        #
        # pad to 30x30
        #

        helper.make_node(
            "Pad",
            ["small_out_fixed"],
            ["output"],
            mode="constant",
            pads=[0,0,0,0,0,0,21,21],
            value=0.0
        )
    ]

    graph = helper.make_graph(
        nodes,
        "task001",
        [x],
        [y],
        inits
    )

   # graph.output.extend([
   #     helper.make_tensor_value_info(
   #         "patch",
   #         TensorProto.FLOAT,
   #         None
   #     ),
   #     helper.make_tensor_value_info(
   #         "kron6",
   #         TensorProto.FLOAT,
   #         None
   #     ),
   #     helper.make_tensor_value_info(
   #         "kron6_t",
   #         TensorProto.FLOAT,
   #         None
   #     ),
   #     helper.make_tensor_value_info(
   #         "small_out",
   #         TensorProto.FLOAT,
   #         None
   #     ),
   # ])

    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[
            helper.make_opsetid("",10)
        ]
    )

    return model

#
#  Save model
#

network = task001_network()

onnx.save(network, "task001.onnx")

session = onnxruntime.InferenceSession("task001.onnx")

bench = convert_to_numpy(examples["train"][0])

inp = bench["input"]
expected = bench["output"]

output, patch, kron6, kron6_t, small = session.run(
    None,
    {"input": inp}
)

#
#  Verify network
#
verify_network(model, task_num, examples)
