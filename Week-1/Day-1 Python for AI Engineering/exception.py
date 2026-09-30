def divide(a, b):
    try:
        result=a/b
    except ZeroDivisionError:
        print("can't divide with 0")
    else:
        print(f"result:{result}")
    finally:
        print("division attempt finished")

divide(10,5)