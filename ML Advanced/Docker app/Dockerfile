FROM python:3.6-slim
COPY ./app.py /app/
COPY ./requirements.txt /app/
COPY ./heart.csv /app/
WORKDIR /app/
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]
