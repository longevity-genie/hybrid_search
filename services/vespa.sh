docker run --name vespa --hostname vespa-container \
  --publish 8081:8080 --publish 19071:19071 \
  vespaengine/vespa