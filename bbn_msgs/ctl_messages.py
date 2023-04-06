

valid_messages = ["start", "stop", "pause"]

class InvalidMessage(Exception):
    pass

class InvalidSkill(InvalidMessage):
    pass

def validate_server_message(message:str) -> bool:
    """makes sure message is in valid format, and raises an exception if it is not"""
    if not any("skill" and s_d in message.lower() for s_d in ['started', 'done']):
        if not message.lower() in valid_messages:
            return False
    elif not any(skill in message.lower() for skill in skills):
        return False
    return True

def get_socket_address(address:str, increase:int=0) -> str:
    """
    Params:
        address: string (e.g. tcp://*:5555)
        increase: int number to increase address port by
    """
    ad = re.sub(r'[0-9]+$',
            lambda x: f"{str(int(x.group())+increase).zfill(len(x.group()))}",address)
    return ad