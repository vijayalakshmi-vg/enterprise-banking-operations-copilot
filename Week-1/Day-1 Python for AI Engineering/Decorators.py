def logger(function):

    def wrapper():
        print("Function started")
        function()
        print("Function finished")

    return wrapper


@logger
def hello():
    print("Hello")


hello()
#2
def log_ticket(function):
    
    def wrapper(text):
        print("processing ticket")
        result=function(text)
        print("ticket processed successfully")
        
        return result
    return wrapper

@log_ticket
def classify_ticket(text):
    if "bill" in text: 
        return "billing" 
    else:
        return "general"

result=classify_ticket("I have a billing problem")
print(result)
    