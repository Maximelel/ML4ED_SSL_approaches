# Use the minimalistic Python Alpine image for smaller size.
FROM python:3.8-slim
# Set the working directory inside the container
WORKDIR /app
# Copy your project code and files into the container
COPY . /app
# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    libgomp1
# Install Python dependencies from requirements.txt
RUN python -m pip install --upgrade pip==23.3.1
RUN pip install -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --no-cache-dir --force-reinstall torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/
#RUN pip install --force-reinstall torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/
#RUN pip install -U huggingface_hub
#RUN pip install -U transformers
#RUN pip install accelerate -U
#RUN pip install wandb
ENV PYTHONPATH=/app

CMD ["python", "train_multiclass_clf_CV.py", "--batch_size", "8", "--epochs", "10", "--n_splits", "5"]