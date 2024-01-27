"""
Copyright 2023 by Raytheon BBN Technologies All Rights Reserved
"""
import zmq
import time
import traceback
import yaml



class UnexpectedResponse(Exception):
    pass

class MissingResponse(Exception):
    pass


skills = ["m1","m2","m3","m5","r18"]
valid_messages = ["start", "stop", "pause"]
ok_message = 'ok'


def parse_message(response: str):
    """split response by ":" to return the sender name and their message.
    """
    name, message = response.split(":", 1) if ':' in response else (response, '')
    return name, message

def validate_message(message: str) -> None:
    """If the message is not "OK", throw an error.
    """
    if message.lower() != ok_message:
        raise UnexpectedResponse("Unexpected response error")

def listen_for_response(n:int, socket: zmq.Socket) -> None:
    """
    the server listens for a response using non-binding recv
    for n seconds.
    """
    for i in range(0,int(n/.2),1):
        try:
            response = socket.recv_string(zmq.NOBLOCK)
            name, message = parse_message(response)
            validate_message(message)
            print(f"{name}:{message}")
            return
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                time.sleep(.2)
                continue
            else:
                traceback.print_exc()
        except UnexpectedResponse as e:
            print(f"UNEXPECTED RESPONSE:{name}:{message}")
            return # received message
    raise MissingResponse(f"No response within {n} seconds")


def try_send(socket, *msgs):
    for msg in msgs:
        try:
            print("sending", msg, '...')
            socket.send_string(msg)
            listen_for_response(5,socket)  # perhaps we should change to use poller??
            print("sent", msg)
        except Exception as e:
            print(type(e).__name__, e)
            return


def run_server(address="tcp://*:5555", interactive=False):
    print(f"Attempting to connect to [{address}]")
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.bind(address)
    print("Bound and online.")
    
    print("Waiting for client...")
    message = socket.recv()  # wait for client
    print("initialized:", message)

    if interactive:
        from IPython import embed
        embed()
        return

    print("Testing experiment messages...")
    for v_m in valid_messages:
        try_send(socket, v_m)
        time.sleep(1)
    
    for skill in skills:
        try_send(socket, f"skill {skill} started", f"skill {skill} done")
        time.sleep(1)

    try_send(socket, f"skill fake_skill started", f"skill fake_skill done")

    print("done :)")



if __name__ == '__main__':
    import fire
    fire.Fire(run_server)
