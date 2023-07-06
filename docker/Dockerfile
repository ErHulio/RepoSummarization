# syntax=docker/dockerfile:1
FROM python:3.10
VOLUME /code
WORKDIR /code
COPY tool/requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN rm requirements.txt
#COPY . .
WORKDIR /code/tool
ENTRYPOINT ["python3", "tool_for_repo_summarization.py"]