class Customer:

    def __init__(self, name, email):
        self.name=name
        self.email=email
        
    def display_info(self):
        return self.name,self.email

cust1=Customer('VG','vg@gmail.com')
print(cust1.display_info())
cust2=Customer('sri','srnad@gmail.com')
print(cust2.display_info())