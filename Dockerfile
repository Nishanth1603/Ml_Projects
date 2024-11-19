FROM  python:3.8-slim-buster
WORKDIR /app
copy . /app

RUN apt update -y && apt install awscli -y
RUN pip install  -r Requirements.txt
CMD ["python3","app.py"]