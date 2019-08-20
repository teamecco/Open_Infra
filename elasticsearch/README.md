# Elasticsearch setting

1. change the setting.json file to your table
2. curl -XPUT elasticsearch_IP:elasticsearch_PORT/[INDEX_NAME]
3. curl -H 'Context-Text application/json' -XPOST elasticsearch_IP:elasticsearch_PORT/[INDEX_NAME]/[TYPE_NAME]
#### [TYPE_NAME] must be the smae name you wrote in the json file mapping
