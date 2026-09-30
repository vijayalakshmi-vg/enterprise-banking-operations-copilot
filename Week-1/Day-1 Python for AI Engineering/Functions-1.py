def calculate(price,tax):
    total_price= price+(price*(tax/100))
    return total_price

print(calculate(100,10))

def classify_ticket(text):
    if "bill" in text:
        print("billing")
    else:
        print("general")

print(classify_ticket("I have a billing problem"))
