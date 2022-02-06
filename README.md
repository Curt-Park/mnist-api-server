# MNIST API Server
MNIST inference server w/ FastAPI

## Prerequisites
```bash
$ make env      # create anaconda environment
$ make setup    # initial setup for the project
```

## How to Play
```bash
$ make train        # train conv net (optional)
$ make backend      # start up the server
$ make frontend     # start up frontend (draw board)
```

## For Developers
```bash
$ make format   # format python scripts
$ make lint     # lint python scripts
$ make utest    # run unit tests
$ make cov      # open coverage report (after `make utest`)
```

## References
- https://github.com/pytorch/examples/blob/master/mnist/main.py
- https://github.com/KiLJ4EdeN/fastapi_tf-keras_example
- https://github.com/rahulsrma26/streamlit-mnist-drawable
- https://github.com/zademn/mnist-mlops-learning
