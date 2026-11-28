FROM rahbar-indexer:f9727ae
WORKDIR /app
ENV PYTHONPATH=/app:/app/eval
COPY eval/ /app/eval/
ENTRYPOINT ["python", "-u", "/app/eval/run_cluster_eval.py"]
