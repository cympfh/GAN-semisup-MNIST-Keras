a:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python main.py train --name $(shell date "+%Y-%m-%d-%s") -l 10

b:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python main.py train --name $(shell date "+%Y-%m-%d-%s") -l 100 -u 500

c:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python main.py train --name $(shell date "+%Y-%m-%d-%s") -l 100 -u 100 --epochs 1000

d:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python main.py train --name $(shell date "+%Y-%m-%d-%s") -l 100 -u 10,10,10,10,10,90,90,90,90,90

e:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python main.py train --name $(shell date "+%Y-%m-%d-%s") -l 100 -u 2000,2000,1000,0,0,0,0,0,0,0

f:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python main.py train --name $(shell date "+%Y-%m-%d-%s") -l 100 -u 5000,1,1,1,1,1,1,1,1,1

## training mode (newly)
train:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python main.py train --name $(shell date "+%Y-%m-%d-%s")

## testing mode
test:
	CUDA_VISIBLE_DEVICES=$(shell empty-gpu-device) python main.py test $(shell ls -1 snapshots/*.h5|peco)

## visplot a log
log:
	bash script/log.sh

.DEFAULT_GOAL := help

## shows this
help:
	@grep -A1 '^## ' ${MAKEFILE_LIST} | grep -v '^--' |\
		sed 's/^## *//g; s/:$$//g' |\
		awk 'NR % 2 == 1 { PREV=$$0 } NR % 2 == 0 { printf "\033[32m%-18s\033[0m %s\n", $$0, PREV }'
