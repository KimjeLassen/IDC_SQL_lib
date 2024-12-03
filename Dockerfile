FROM python:3.12.7

WORKDIR /idc-app

COPY . /idc-app

RUN pip3 install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "app.Main:app", "--host", "0.0.0.0", "--port", "8000"]