docker run \
    --name predict \
    --rm \
    -v $(pwd)/dtu_mlops_cookiecutter_example/models/model.pt:/dtu_mlops_cookiecutter_example/models/model.pt \
    -v $(pwd)/data/np_image_test.npy:/data/np_image_test.npy \
    lucasmogsan/predict:latest \
        /dtu_mlops_cookiecutter_example/models/model.pt \
        /data/np_image_test.npy