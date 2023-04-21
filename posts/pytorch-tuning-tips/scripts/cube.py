import os
import argparse
from subprocess import run, CalledProcessError

cwd = os.getcwd()


docker_images = {
    'pytorch-ngc:22.11-py3': {'container-src': 'ngc'},
    # 'pytorch-ngc:22.12-py3': {'container-src': 'ngc'},
    # 'pytorch-ngc:23.01-py3': {'container-src': 'ngc'},
    'pytorch-ngc:23.02-py3': {'container-src': 'ngc'},
    'pytorch-ngc:23.03-py3': {'container-src': 'ngc'},
    # 'pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime': {'container-src': 'pytorch'},
    # 'pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime': {'container-src': 'pytorch'},
    # 'pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime': {'container-src': 'pytorch'},
    'pytorch-hub:1.13.1-cuda11.6-cudnn8-runtime': {'container-src': 'pytorch'},
    'pytorch-hub:2.0.0-cuda11.7-cudnn8-runtime': {'container-src': 'pytorch'},
    'pytorch2-nightly:latest': {'container-src': 'pytorch'},
}

model_arch = ['bert-base-uncased', 'bert-large-uncased', 'distilgpt2', 'resnet18', 'resnet50']

def bool_csv(s):
    return [t.strip().lower() == 'true' for t in s.split(',')]

def str_csv(s):
    return [str(t.strip()) for t in s.split(',')]

a = argparse.ArgumentParser()
a.add_argument('--docker', default=docker_images.keys(), nargs='*')
a.add_argument('--model-arch', default=model_arch, nargs='*')
a.add_argument('--inference-mode', default=[True, False], type=bool_csv)
a.add_argument('--no-grad', default=[False], type=bool_csv)
a.add_argument('--cudnn-benchmark', default=[True, False], type=bool_csv)
a.add_argument('--autocast-dtype', default=['float32', 'float16'], type=str_csv)
a.add_argument('--channels-last', default=[True, False], type=bool_csv)
a.add_argument('--eval-mode', default=[True, False], type=bool_csv)
a.add_argument('--compile', default=[True, False], type=bool_csv)
args = a.parse_args()


def run_bench(docker_image, params, switches):
    print(docker_image, params, switches)
    try:
        run([
            'docker', 'run', '-it', '--rm', '--gpus', 'all', '--ipc=host', '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864',
            '-v', f'{cwd}:/workspace',
            '-v', '/home/pbridger/dev/paulbridger.com/cache:/root/.cache',
            '-v', f'{cwd}/entrypoint.d:/opt/nvidia/entrypoint.d',
            docker_image,
            'python',  'bench.py', '--csv-path=next.csv',
            *[f'--dim={k}={v}' for k, v in dims.items()],
            *[f'--{k}={v}' for k, v in params.items()],
            *[s for s in switches if s]
        ], check=True)
    except CalledProcessError as exc:
        print(exc)


for docker_image in args.docker:
    dims = docker_images[docker_image]
    for model_arch in args.model_arch:
        for inference_mode in args.inference_mode:
            for no_grad in args.no_grad:
                for cudnn_benchmark in args.cudnn_benchmark:
                    for autocast_dtype in args.autocast_dtype:
                        for channels_last in args.channels_last:
                            for eval_mode in args.eval_mode:
                                for pyt2_compile in args.compile:
                                    # for set_to_none in [False, True]:
                                    run_bench(
                                        docker_image, {
                                            'model-arch': model_arch,
                                        }, [
                                            '--inference-mode' if inference_mode else '',
                                            '--no-grad' if no_grad else '',
                                            '--cudnn-benchmark' if cudnn_benchmark else '',
                                            f'--autocast-dtype={autocast_dtype}' if autocast_dtype else '',
                                            '--channels-last' if channels_last else '',
                                            '--eval' if eval_mode else '',
                                            '--compile' if pyt2_compile else '',
                                            # '--set-to-none' if eval_mode else '',
                                        ]
                                    )



