"""
Copyright 2023 by Raytheon BBN Technologies All Rights Reserved

# Server side code for client-server messaging system
# Uses 0mq messaging
#
# Binds socket at either given or default address, then
# waits to receive message from an attached client.
# Sends confirmation message after receiving message.
"""
import zmq
import yaml



def parse_message(message):
    # convert message to YAML object
    yaml_object = yaml.safe_load(message)      
    for step in yaml_object['users current actions right now']['steps']:
        #print out the current step for the user
        if step['state'] == 'current':
            print("New Message Timestamp: " + str(yaml_object['header']['transmit timestamp']))
            print(step)
            print('\n') 
        

def run_server(address="tcp://*:5555"):
    print("Attempting to connect to [" + address + "]")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(address)
    print("Bound and online...")

    while True:
        # Wait for next message from client
        message = socket.recv_string()
        parse_message(message)
        socket.send_string("Message Received")


if __name__ == '__main__':
    import fire
    fire.Fire(run_server)