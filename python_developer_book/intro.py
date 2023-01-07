# This is from Dive Into Design Patterns

# Basics of OOP

'''
OOP is a paradigm on the concept of wrapping data + related behaviour
into special bundles called `objects`. The `objects` are constructed
through blueprints called `classes`.

Generally, the object Nalle is created by the class Cat. And a class
contains its name, fields/state and methods. 
'''

class Cat():

    # The state is specified by variables in self
    def __init__(self, **kwargs):
        self.name = kwargs.name
        self.gender = kwargs.gender
        # ...
    
    # Methods defined in class
    def breathe():
        pass
    
    def eat(food):
        print(food)

class Nalle(Cat):

    # Inherit the basic cat methods + specialized ones
    def bite_for_no_reason():
        pass