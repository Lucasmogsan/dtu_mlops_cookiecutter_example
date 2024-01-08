# Docker

How to build a docker file for our MNIST repository that will make the training and prediction a self contained application.

## How to build image
The ```-f train.dockerfile .``` (the dot is important to remember as it is the dir we build from) indicates which dockerfile that we want to run (except if you named it just ```Dockerfile```) and the ```-t trainer:latest``` is the respective name / tag that we se afterwards when running ```docker images```.
```bash
cd <root>
docker build -f dockerfiles/trainer.dockerfile . -t trainer:latest
```


## How to run

To give our container name ```experiment1``` using the image ```trainer:latest```. We can also run it interactively (```-it```), from another entrypoint (```--entrypoint```) and/or with a volume mounted for a shared folder (e.g. to share the model and outputs between local and container).
```bash
docker run --name {desired_container_name} {image_name}
docker run --name experiment1 trainer:latest
docker run -it --entrypoint sh {image_name}:{image_name}
docker run -it --entrypoint sh trainer:latest
docker run --name experiment1 -v $(pwd)/dtu_mlops_cookiecutter_example/models:/dtu_mlops_cookiecutter_example/models/ trainer:latest
```


For the **prediction** dockerimage we need some extra arguments to mount and pass directiories (remove commets to make it work):
```bash
docker run --name predict --rm \    # --rm removes the container after it's done
    -v $(pwd)/dtu_mlops_cookiecutter_example/models/model.pt:/dtu_mlops_cookiecutter_example/models/model.pt \  # mount trained model file
    -v $(pwd)/data/np_image_test.npy:/data/np_image_test.npy \  # mount data we want to predict on
    predict:latest \    # image name
    /dtu_mlops_cookiecutter_example/models/model.pt \   # argument(s) to script, path relative to dir of execution.
    /data/np_image_test.npy
```

A **shell-script** can be benficial to create to run the container when it becomes more complicated with multiple inputs. <br />
In this you can also add arguments - e.g. for a shared folder. You might need to giver permission to the file first.
```bash
chmod +rwx docker-run-predict.sh
./docker-run-predict.sh
```

A **docker compose file** is somewhat similar to the shell script but can contain multiple docker commands.





**Start** a container (```-i``` for interativate) - obs. starts from entrypoint unless
```bash
docker start {container_name}
docker start experiment1
```
**Open** more terminals for a container
```bash
docker exec -it {container_name} bash
docker exec -it experiment1 bash
```



## (optional) Push and pull image to/from dockerhub
1. Rename the tag to include the link to ```docker.io``` and your dockerhub username ```$DOCKERHUB_USER```:
```bash
docker tag trainer:latest docker.io/$DOCKERHUB_USER/trainer:latest
docker tag trainer:latest docker.io/lucasmogsan/trainer:latest
```

2. Push to dockerhub:
```bash
docker push docker.io/$DOCKERHUB_USER/trainer:latest
docker push docker.io/lucasmogsan/trainer:latest
```

3. Verify that it is uploaded to [https://hub.docker.com/](https://hub.docker.com/)

4. Pull image
```bash
docker pull $DOCKERHUB_USER/trainer:latest
docker pull lucasmogsan/trainer:latest
```


## Other commands:
```bash
docker system prune -a
docker exec -it
docker cp {container_name}:{dir_path}/{file_name} {local_dir_path}/{local_file_name}
```