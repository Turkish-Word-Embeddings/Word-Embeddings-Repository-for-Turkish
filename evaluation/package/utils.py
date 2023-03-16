
def get_yes_no_input(message=None):
    if message:
        print(message)
    response = None
    while response not in ["yes", "no"]:
        response = input(" -> yes/[no]: ") or "no"
    return response