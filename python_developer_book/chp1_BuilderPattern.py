# Packages
import os
import sys
import tempfile
import abc  # abstract base classes
if sys.version_info[:2] < (3, 2):
    from xml.sax.saxutils import escape
else:
    from html import escape

# Note: The main goal of the abstract base class is to provide a standardized 
# way to test whether an object adheres to a given specification.

# Doesn't seem that widely used tbh. But overall concept is Abstract Class
# vs Duck-Typing. Abstract is the idea that Duck() and Mallard() both
# inherit from abstract Animal() and it's abstract methods like 
# Animal().eat(). But Animal() itself will never be instantiated on its own.
#
# Duck-typing on the other hand says that if it walks & quacks like a duck,
# it's a duck. So if there is a .quack() method, it doesn't matter if it
# originates from Duck() or Mallard(). 

# -------------------------------------------------- #
# -------------------------------------------------- #
# Builder Pattern (Creational Pattern)

# The Builder Pattern is similar to the Abstract Factory Pattern in that both
# patterns are designed for creating complex objects composed of smaller
# less complex ones. The difference is that the main builder class not
# only provides the methods for building a complex object, but it holds
# the representation of the entire complex object itself. 

# formbuilder.py, this is pseduo code
# We have a create_login_form() function to process different types
# of login form classes and write to a file.

def main():
    htmlFilename = os.path.join(tempfile.gettempdir(), "login.html")
    htmlForm = create_login_form(HtmlFormBuilder())
    with open(htmlFilename, 'w', encoding='utf-8') as file:
        file.write(htmlForm)

    tkFilename = os.path.join(tempfile.gettempdir(), "login.py")
    tkForm = create_login_form(TkFormBuilder())
    with open(tkFilename, 'w', encoding='utf-8') as file:
        file.write(tkForm)

# Define the function for processing form
# It takes a class for html, tkinter and other additional forms
def create_login_form(builder):
    builder.add_title("Login")
    builder.add_label("Username", 0, 0, target="username")
    builder.add_entry("username", 0, 1)
    builder.add_label("Password", 1, 0, target="password")
    builder.add_entry("password", 1, 1, kind="password")
    builder.add_button("Login", 2, 0)
    builder.add_button("Cancel", 2, 1)
    return builder.form()

# Define abstract form builder that html and tkinter can inherit
# metaclass=abc.ABCMeta means we can never instantiate
# AbstractFormBuilder() directly.
class AbstractFormBuilder(metaclass=abc.ABCMeta):

    # What abstractmethod does is that it prevents you from 
    # instantiating builder() without a method like add_title().
    @abc.abstractmethod
    def add_title(self, title):
        self.title = title
    
    @abc.abstractmethod
    def form(self):
        pass

    # ....

class HtmlFormBuilder(AbstractFormBuilder):

    def __init__(self):
        self.title = "HtmlFormBuilder"
        self.items = {}
    
    def add_title(self, title):
        # super() to inherit method from the superclass 
        # superclass is anything above the subclass
        super().add_title(escape(title))

    # ...

class TkFormBuilder(AbstractFormBuilder):

    def __init__(self):
        self.title = "TkFormBuilder"
        self.statements = []


    def add_title(self, title):
        super().add_title(title)
    
    # ...


# Discussion: This form provides the same compositionality as the
# Abstract Factory Pattern. The difference is that it's well suited
# for problems where we want the builder to not only provider
# the methods, but also hold the representation of the complex object.
# In this case, a builder e.g. HtmlFormBuilder() holds all the methods,
# but it also directly 

# Framework: This pattern is useful when you want to construct complex 
# objects step by step. The pattern allows you to produce different 
# types and representations of an object using the same construction code.
# So ideally, you want a process that has similar construction steps.

# Example: A class to build houses. If we create a subclass for each type
# of housing, it'll get ugly very fast. E.g. condo, townhouse, house,
# no swimming pool, extra closet, balcony etc.. If we try to create
# one very extensive function, it'll end up with a lot of arguments set
# to false and again, very ugly.
# The solution is to create a HouseBuilder() base class and add
# methods for each feature, e.g. buildWalls(), buildWindows(), buildPools()
# that are optional to call. 
# Moreover, we can add HouseDirector() that specifies the flow of these
# optional functions, so it can direct the building process when given
# a specific command e.g. modern condo.