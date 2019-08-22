import threading
import pika
import global_data

class Threaded_consumer(threading.Thread):
    def callback(self, ch, method, properties, body):
        global_data.q.put(body)

    def __init__(self):
        threading.Thread.__init__(self)

        self.HOST = '106.10.38.29'
        self.PORT = 5672
        self.Virtual_Host = '/'
        self.credentials = pika.PlainCredentials('admin', 'admin')
        self.parameters = pika.ConnectionParameters(self.HOST, self.PORT,self.Virtual_Host, self.credentials)
        self.connection = pika.BlockingConnection(self.parameters)
        self.channel = self.connection.channel()

        self.channel.basic_consume(on_message_callback=self.callback,queue='sensor', auto_ack=True)

    def run(self):
        print('start consuming')
        self.channel.start_consuming()

