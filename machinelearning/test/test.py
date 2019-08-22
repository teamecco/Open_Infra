from elasticsearch import Elasticsearch
from elasticsearch import helpers
import json
import time

es = Elasticsearch("http://106.10.51.176:31032/")

#============search query======================================
query = { # one minute data
    "query":{
        "range": {
            "@timestamp":{
                "gte":"now-2s/s",
                "lt":"now"
            }
        }
    },
    "_source": ['pi1_temp', 'pi1_cpu', 'pi1_ram', '@timestamp'],
}

while True:
    results = es.search(index='sensor',body=query)
    timestamp = ''
    for msg in results['hits']['hits']:
        if not(timestamp == msg['_source']['@timestamp']) :
            input = [msg['_source']['pi1_temp'],msg['_source']['pi1_cpu'],msg['_source']['pi1_ram'],msg['_source']['@timestamp']]
            print(input)
            timestamp = msg['_source']['@timestamp']
    time.sleep(2)

#input query


