IMG := dsb-2019
TAG := latest
USER := kaggle


docker-build:
	docker build --pull --rm -t ${IMG}:${TAG} .

env:
	docker run --rm -it --init \
	--ipc host \
	--name dsb2019 \
	--volume `pwd`:/app/dsb2019 \
	-w /app/dsb2019 \
	--user `id -u`:`id -g` \
	--publish 9000:9000 \
	${IMG}:${TAG} /bin/bash

jupyter:
	sudo chown ${USER}:${USER} /home/user/.jupyter
	jupyter lab --port 9000 --ip 0.0.0.0 --NotebookApp.token=
