import json
from time import time
from time import sleep
from random import random
from flask import Flask, render_template, make_response
import elasticsearch

app = Flask(__name__)

es = elasticsearch.Elasticsearch("http://106.10.51.176:31032/")

#============search query======================================
query = { # one minute data
    "query":{
        "range": {
            "@timestamp":{
                "gte":"now-1s/s",
                "lt":"now"
            }
        }
    },
    "_source": ['pi1_temp', 'pi1_cpu', 'pi1_ram', '@timestamp'],
}


@app.route('/')
def hello_world():
    global data
    return render_template('index0.html')

@app.route('/live-data')
def live_data():
    # Create a PHP array and echo it as JSON
    #data = [time() * 1000, random() * 100]
    while True:
        results = es.search(index='sensor',body=query)
        print(results)
        num = results['hits']['total']
        print(type(num))
        print(num)
#    data = []
        if not(num == 0) :
            print("hi")
            for msg in results['hits']['hits']:
                data = [time() * 1000,msg['_source']['pi1_ram'],msg['_source']['pi1_temp']]
#        if not(timestamp == msg['_source']['@timestamp']) :
            #        data = [msg['_source']['pi1_temp'],msg['_source']['pi1_cpu'],msg['_source']['pi1_ram']]
            print(data)
            response = make_response(json.dumps(data))
            response.content_type = 'application/json'
            return response
        sleep(0.5)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5049)
