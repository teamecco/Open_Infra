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
    return render_template('index0.html')

@app.route('/cpu')
def return_cpu():
    return render_template('index0.html')

@app.route('/temp')
def return_temp():
    return render_template('index1.html')
    
@app.route('/ram')
def return_ram():
    return render_template('index2.html')

@app.route('/live-temp')
def live_temp():
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
                data = [time() * 1000,msg['_source']['pi1_temp']]
#        if not(timestamp == msg['_source']['@timestamp']) :
            #        data = [msg['_source']['pi1_temp'],msg['_source']['pi1_cpu'],msg['_source']['pi1_ram']]
            print(data)
            response = make_response(json.dumps(data))
            response.content_type = 'application/json'
            return response
        sleep(0.5)

@app.route('/live-cpu')
def live_cpu():
    # Create a PHP array and echo it as JSON
    #data = [time() * 1000, random() * 100]
    while True:
        results = es.search(index='sensor',body=query)
        print(results)
        num = results['hits']['total']
        print(type(num))
        print(num)
        if not(num == 0) :
            print("hi")
            for msg in results['hits']['hits']:
                data = [time() * 1000,msg['_source']['pi1_cpu']]
#        if not(timestamp == msg['_source']['@timestamp']) :
            #        data = [msg['_source']['pi1_temp'],msg['_source']['pi1_cpu'],msg['_source']['pi1_ram']]
            print(data)
            response = make_response(json.dumps(data))
            response.content_type = 'application/json'
            return response
        sleep(0.5)

@app.route('/live-ram')
def live_ram():
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
                data = [time() * 1000,msg['_source']['pi1_ram']]
#        if not(timestamp == msg['_source']['@timestamp']) :
            #        data = [msg['_source']['pi1_temp'],msg['_source']['pi1_cpu'],msg['_source']['pi1_ram']]
            print(data)
            response = make_response(json.dumps(data))
            response.content_type = 'application/json'
            return response
        sleep(0.5)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5050)
