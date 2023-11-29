import pika
from utils import exponential_backoff
import logging

class RMQClient:
	def __init__(self, server, port, user, password, virtual_host='/'):
		self.server = server
		self.port = port
		self.user = user
		self.password = password
		self.virtual_host = virtual_host
		self.connection = None
		self.channel = None
		self.reconnect_count = 0  # Delay in seconds before attempting to reconnect

	def __connect(self):
		creds = pika.PlainCredentials(self.user, self.password)
		connection_params = pika.ConnectionParameters(host=self.server,
											port=self.port,
											virtual_host=self.virtual_host,
											credentials=creds)
		try:
			self.connection = pika.BlockingConnection(connection_params)
			self.channel = self.connection.channel()
		except pika.exceptions.AMQPConnectionError as e:
			logging.error(f"Error connecting to RabbitMQ: {e}")
			self.__reconnect()

	def __reconnect(self):
		_ = exponential_backoff(self.reconnect_count)
		self.reconnect_count += 1
		self.__connect()  # Attempt to reconnect

	def publish_exchange(self, exchange, body, routing_key=''):
		if not self.channel:
			self.__connect()
		try:
			self.channel.basic_publish(exchange=exchange, body=body, routing_key=routing_key)
			logging.info(f"Message published: {body} to exchange {exchange} with routing_key {routing_key}")
		except (pika.exceptions.AMQPConnectionError, pika.exceptions.ChannelClosedByBroker) as e:
			logging.error(f"Publishing error, attempting reconnect: {e}")
			self.__reconnect()
			self.publish_exchange(exchange, body, routing_key)

	def process(self, callback_on_message, source_queue, completed_exchange):
		if not self.channel:
			self.__connect()
		try:
			self.channel.basic_consume(queue=source_queue, on_message_callback=callback_on_message, auto_ack=True)
			logging.info("Waiting for messages. To exit press CTRL+C")
			self.channel.start_consuming()
		except KeyboardInterrupt:
			self.channel.stop_consuming()
			self.connection.close()
			logging.info("Closing connection.")
		except (pika.exceptions.AMQPConnectionError, pika.exceptions.ChannelClosedByBroker) as e:
			logging.error(f"Error in consumption: {e}, attempting to reconnect...")
			self.__reconnect()
			self.process(callback_on_message, source_queue, completed_exchange)
