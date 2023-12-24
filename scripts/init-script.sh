#!/bin/bash
# init-script.sh

# Wait for OpenSearch to be online
while ! curl -XGET 'http://opensearch-node1:9200/_cluster/health?wait_for_status=yellow&timeout=50s' ; do
  sleep 10
done

# Create index patterns
# You'd replace this with the actual API calls to create your index patterns
curl -X POST "opensearch-node1:9200/_index_template/my_index_pattern" -H 'Content-Type: application/json' -d'
{
  "index_patterns": ["index_*"],
  "template": {
    "settings": {
      "number_of_shards": 1
    },
    "mappings": {
      "properties": {}
    }
  }
}
'