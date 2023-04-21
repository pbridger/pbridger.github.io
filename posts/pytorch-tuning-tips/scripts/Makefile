

pytorch2-nightly: docker/pytorch2-nightly.Dockerfile
	docker build -t pytorch2-nightly -f $< .

pytorch-hub-%: docker/pytorch-ngc.Dockerfile
	docker build -t pytorch-hub:$* -f $< --build-arg BASE=pytorch/pytorch:$* .

plotly: docker/plotly.Dockerfile
	docker build -t $@ -f $< .

pytorch-ngc-%: docker/pytorch-ngc.Dockerfile
	docker build -t pytorch-ngc:$* -f $< --build-arg BASE=nvcr.io/nvidia/pytorch:$* .

ngc-containers: pytorch-ngc-22.11-py3 pytorch-ngc-22.12-py3 pytorch-ngc-23.01-py3 pytorch-ngc-23.02-py3
hub-containers: pytorch-hub-2.0.0-cuda11.7-cudnn8-runtime pytorch-hub-1.13.1-cuda11.6-cudnn8-runtime


DRUN=@docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(shell pwd):/workspace -v ~/dev/paulbridger.com/cache:/root/.cache -v $(shell pwd)/entrypoint.d:/opt/nvidia/entrypoint.d 

sh-plotly: plotly
	$(DRUN) \
		plotly

sh-ngc:
	$(DRUN) \
		pytorch-ngc:22.11-py3

sh-ngc2:
	$(DRUN) \
		pytorch-ngc:23.03-py3

sh-nightly:
	$(DRUN) \
		pytorch2-nightly

sh-hub:
	$(DRUN) \
		pytorch-hub:2.0.0-cuda11.7-cudnn8-runtime


cube:
	rm -f next.csv
	python cube.py


test-ngc:
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast --eval
	$(DRUN) \
		pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime \
		python bench.py --autocast --eval
	# $(DRUN) \
	# 	pytorch-ngc:23.01-py3 \
	# 	python bench.py --autocast
	# $(DRUN) \
	# 	pytorch-ngc:23.02-py3 \
	# 	python bench.py --autocast


test-pytorch:
	$(DRUN) \
		pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime \
		python bench.py --model-arch=resnet50
	$(DRUN) \
		pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime \
		python bench.py --model-arch=resnet50
	$(DRUN) \
		pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime \
		python bench.py --model-arch=resnet50


test-inference-mode:
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --batch-size=1
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --no-grad --batch-size=1
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --inference-mode --batch-size=1


test-set-to-none:
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --optimize --batch-size=1
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --optimize --set-to-none --batch-size=1


test-cudnn-benchmark:
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --optimize
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --cudnn-benchmark --optimize
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast --optimize
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast --cudnn-benchmark --optimize
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --cudnn-benchmark
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast --cudnn-benchmark


test-autocast:
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast
	# >= Ampere
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast --autocast-bf16


test-channels-last:
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --channels-last --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast --cudnn-benchmark
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --channels-last --autocast --cudnn-benchmark


test-lazy:
	$(DRUN) -e CUDA_MODULE_LOADING=lazy \
		pytorch-ngc:22.11-py3 \
		python bench.py
	$(DRUN) -e CUDA_MODULE_LOADING=eager \
		pytorch-ngc:22.11-py3 \
		python bench.py


test-pessimization:
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast --pessimization


test-eval:
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast-dtype=float16 --csv-path=test.csv
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast-dtype=float16 --eval --csv-path=test.csv


test-bnb:
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --optimize --model-arch=resnet50
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --optimize --model-arch=resnet50 --bnb
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast --optimize --model-arch=resnet50
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast --optimize --model-arch=resnet50 --bnb


test-compile:
	$(DRUN) \
		pytorch2-nightly \
		python bench.py --autocast
	$(DRUN) \
		pytorch2-nightly \
		python bench.py --autocast --compile
	$(DRUN) \
		pytorch2-nightly \
		python bench.py --autocast --optimize
	$(DRUN) \
		pytorch2-nightly \
		python bench.py --autocast --optimize --compile


test-jit-script:
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --jit-script --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --autocast --batch-size=1
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --jit-script --autocast --batch-size=1


test-resolution:
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --batch-size=1 --resolution=384 --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --batch-size=2 --resolution=384 --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --batch-size=3 --resolution=384 --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --batch-size=4 --resolution=384 --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --batch-size=5 --resolution=384 --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --batch-size=6 --resolution=384 --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --batch-size=7 --resolution=384 --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --batch-size=8 --resolution=384 --autocast
	$(DRUN) \
		pytorch-ngc:22.11-py3 \
		python bench.py --batch-size=9 --resolution=384 --autocast


