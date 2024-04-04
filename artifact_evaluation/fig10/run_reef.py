import os
import time

num_runs = 1
# trace_files = [
#     ("ResNet50", "ResNet50", "rnet_rnet", 160000),
#     ("ResNet50", "MobileNetV2", "rnet_mnet", 100000),
#     ("MobileNetV2", "ResNet50", "mnet_rnet", 160000),
#     ("MobileNetV2", "MobileNetV2", "mnet_mnet", 100000),
#     ("ResNet101", "ResNet50", "rnet101_rnet", 160000),
#     ("ResNet101", "MobileNetV2", "rnet101_mnet", 100000),
#     ("BERT", "ResNet50", "bert_rnet", 160000),
#     ("BERT", "MobileNetV2", "bert_mnet", 100000),
#     ("Transformer", "ResNet50", "trans_rnet", 160000),
#     ("Transformer", "MobileNetV2", "trans_mnet", 100000),
# ]

trace_files = [
    ("ResNet50", "ResNet50", "rnet_rnet"),
]

for (be, hp, f) in trace_files:
    for run in range(num_runs):
        print(be, hp, run, flush=True)
        # run
        file_path = f"config_files/{f}.json"
        os.system(f"LD_PRELOAD='{os.path.expanduser( '~' )}/orion/src/cuda_capture/libinttemp.so' python3.10 ../../benchmarking/launch_jobs.py --algo reef --config_file {file_path} --reef_depth 12")

        # copy results
        # os.system(f"cp client_1.json results/reef/{be}_{hp}_{run}_hp.json")
        # os.system("rm client_1.json")
