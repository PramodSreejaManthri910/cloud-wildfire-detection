# AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.12

# Copy dependency list and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy Lambda inference code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# The trained model must be placed in models/
# before building the deployment image.
COPY models/wildfire_model.keras ${LAMBDA_TASK_ROOT}/models/wildfire_model.keras
COPY models/class_indices.json ${LAMBDA_TASK_ROOT}/models/class_indices.json

# AWS Lambda entry point
CMD ["lambda_function.lambda_handler"]
