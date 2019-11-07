IMG := dsb-2019
TAG := latest


docker-build:
	docker build --pull --rm -t ${IMG}:${TAG} .

env:
	docker run --rm -it --init \
	--ipc host \
	--name dsb2019 \
	--volume `pwd`:/app/dsb2019 \
	-w /app/dsb2019 \
	--user `id -u`:`id -g` \
	${IMG}:${TAG} /bin/bash
