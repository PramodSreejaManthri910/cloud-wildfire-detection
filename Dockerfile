# AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.12

# Copy dependency list
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python dependencies
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy trained wildfire model
COPY models/wildfire_model.keras ${LAMBDA_TASK_ROOT}/models/wildfire_model.keras

# Copy Lambda inference function
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# AWS Lambda handler
CMD ["lambda_function.lambda_handler"]
