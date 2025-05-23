# set the base image
FROM python:3.12-slim

# install lightgbm dependency
RUN apt-get update && apt-get install -y libgomp1

# set up the working directory
WORKDIR /app

# copy the requirements file
COPY requirements-dockers.txt ./

# install the packages
RUN pip install -r requirements-dockers.txt
RUN pip install streamlit
# copy the app contents
COPY app.py ./
COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./scripts/data_clean_utils.py ./scripts/data_clean_utils.py
COPY ./run_information.json ./
COPY frontend.py ./

# expose the port
EXPOSE 8000 8501

# Run the file using command
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 & \
                  streamlit run frontend.py --server.port=8501 --server.address=0.0.0.0"]