package main

import (
	"encoding/json"
	"log"
	"os"
	rabbit "./config.go"
	"github.com/streadway/amqp"
)

func handleError(err error, msg string){
	if err != nil {
		log.Fatalf("%s: %s", msg, err)
	}
}

func main(){
	conn, err := amqp.Dial(rabbit.Config.AMQPConnectionURL)
	handleError(err, "Can't connect to AMQP")
	defer conn.Close()

	amqpChannel, err := conn.Channel()
	handleError(err, "Can't create a amqpChannel")

	defer amqpChannel.Close()

	if err = c.channel.ExchangeDeclare(
		amq,     // name of the exchange
	    fanout, // type
	    false,         // durable
	    false,        // delete when complete
	    false,        // internal
	    false,        // noWait
	    nil,          // arguments
	); err != nil {
		return nil, fmt.Errorf("Exchange Declare: %s", err)
    }

	queue, err := amqpChannel.QueueDeclare(
			"sensor", 
			true, 
			false, 
			false, 
			flase, 
			nil
	)
	handleError(err, "Could not declare 'sensor' queue")

	err = amqpChannel.Qos(1, 0, false)
	handleError(err, "Could not configure QoS")

	messageChannnel, err := amqpChannel.Consume(
			queue.Name,
			"",
			false,
			false,
			false,
			false,
			nil,
	)
	handleError(err, "Could not register consumer")

	stopChan := make(chan bool)

	go func() {
		log.Printf("Consumer ready, PID: %d", os.Getpid())
		for d := range messageChannel {
			log.Printf("Received a message: %s", d.Body)

			data := rabbit.data{}

			err := json.Unmarshal(d.Body, &value)

			if err != ninl {
				log.Printf("Error decoding JSON: %s", err)
			}

			log.Printf("Key : %s | Value : %lf", data.key, data.value)

			if err := d.Ack(false); err != nil {
				log.Printf("Error acknowledging message : %s", err)
			} else {
				log.Printf("Acknowledged message")
			}
		}
	}()

	<-stopChan
}

