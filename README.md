## Gillis
A serverless-based ML model serving framework with automatic model partitioning


### 1. Intro
For a large DNN model, Gillis can divide it into multiple partitions using two partitioning algorithms, latency-optimal and SLO-aware, then automatically deploy model partitions on serverless platforms, including AWS Lambda, Google Cloud Functions and KNIX.

Currently, Gillis supports ONNX models and MXNet runtime.

### 2. Toy example
#### 2.1 Prepare a model

```bash
cd partition
# download vgg-16
wget https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.onnx
mkdir -p models
mv vgg16.onnx models/
```
#### 2.2 Partition a model using latency-optimal scheme

```bash
python main.py lo -n vgg16.onnx -p true 
```
#### 2.3 Deploy partitions on AWS Lambda

First, we copy the generated model partitions to the deployment directory, e.g., aws_lambda_deploy.

```
cd ..
mv vgg16_workspace/ aws_lambda_deploy/
```

Then, we deploy partitions on AWS Lambda.

```bash
cd aws_lambda_deploy
bash deploy.sh -j vgg16_workspace
```

Then you can follow the guides of `aws-sam` to finish the deployment.

#### 2.4 After deployment

If everything is going well, you can see an API for model inference. Copy it and try the following command out!

```bash
curl [API]
```