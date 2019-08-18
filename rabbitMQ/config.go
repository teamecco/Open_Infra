package rabbitMQ

type Configuration struct {
		AMQPConnectionURL string
}

type data struct {
	key string
	value float32
}

var Config = Configuration{
		AMQPConnectionURL: "amqp://guest:guest@106.10.38.29:5672/",
}
