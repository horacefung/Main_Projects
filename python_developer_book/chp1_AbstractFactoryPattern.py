# Introduction & Chapter 1
# https://refactoring.guru/design-patterns/creational-patterns

# Packages
import os
import sys
import tempfile

# Chapters:
# 1: Creational Design Patterns
# 2: Structural Design Patterns
# 3: Behavioural Design Patterns
# 4: High-Level Concurrency
# 5: Extending Python
# 6: High-Level Networking
# 7: Graphical User Interfaces (Tkinter)
# 8: OpenGL 3D Graphics


# Overview
# The book has 4 key themes: design patterns for coding elegance,
# improved processing speeds using concurrency + compiled Python (Cython),
# high-level networking and graphics. (We probably skip graphics)

# Some high-level concepts and implementations to keep in mind as we dive
# into this book: 
# 
# i) Knowing different design is crucial to be able to design
# elegeant and reusable code, but some design patterns are irrelevant to most
# use-cases
# 
# ii) CPU-bound processing can be sped up with `multiprocessing` module.
# IO-bound processing can also use `multiprocessing`, or we can do use `threading`
# or `concurrent.futures`. When we need to go to lower/medium levels of concurrency
# we can use `queue`, `multiprocessing` and `concurrent.futures` to minimize errors.
#
# iii) We can also speed up by using Python modules that are written in C under-the-hood.
# We can also use PyPy Python Interpreter which has a just-in-time compiler. Lastly, we
# can we cProfile to discover bottlenecks and write speed-critical code in Cython.


# Chapter 1:
# Creational design patterns are concerned with how objects are created.

# -------------------------------------------------- #
# -------------------------------------------------- #
# Abstract Factory Pattern (Creational Pattern)

# Creational Patterns: Provide mechanisms for creating objects. 
# Ideal when you expect to extend and create new objects as the codebase grows.

# This is for when we want to create complex objects that are made of other objects and these
# other objects belong to one particular family. For example, an abstract widget factory that
# has 3 concreate subclass factories doing the same thing, but for Mac, Xfce and Windows.
# Hence, we just need an abstract factory object, then define a function to convert the
# outputs to Mac/Xfce/Windows.
#
# diagram1.py real code, pseudo code below

# main() and create_diagram() are common functions for both
# svg and txt. The inputs are different based on DiagramFactory()
# or SvgDiagramFactory() but uses the same create_diagram() process.
def main():
    textFilename = os.path.join(tempfile.gettempdir(), "diagram.txt")
    svgFilename = os.path.join(tempfile.gettempdir(), "diagram.svg")

    txtDiagram = create_diagram(DiagramFactory())
    txtDiagram.save(textFilename)

    svgDiagram = create_diagram(SvgDiagramFactory())
    svgDiagram.save(svgFilename)

def create_diagram(factory):
    diagram = factory.make_diagram(30, 7)
    rectange = factory.make_rectangle(4, 1, 22, 5, 'yellow')
    text = factory.make_text(7, 3, 'Abstract Factory')
    #diagram.add(rectangle)
    diagram.add(text)
    return diagram

# Here we then define the specific concrete subclasses
class DiagramFactory:
    def make_diagram(self, width, height):
        return Diagram(width, height)
    
    #def make_recatangle(self, x, y, width, height, fill='white', stroke='black'):
    #    return Rectangle(x, y, width, height, fill, stroke)
    
    def make_text(self, x, y, text, fontsize=12):
        return Text(x, y, text, fontsize)

class SvgDiagramFactory(DiagramFactory):
    def make_diagram(self, width, height):
        return SvgDiagram(width, height)

class Text:
    def __init__(self, x, y, text, fontsize):
        self.x = x
        self.y = y
        self.rows = [list(text)]

class Diagram:
    def add(self, component):
        for y, row in enumerate(component.rows):
            for x, char in enumerate(row):
                self.diagram[x + component.y][x + component.x] = char

SVG_TEXT = """<text x="{x}" y="{y}" text-anchor="left" \ font-family="sans-serif" font-size="{fontsize}">{text}</text>"""
SVG_SCALE = 20 
class SvgText:
    def __init__(self, x, y, text, fontsize): 
        x *= SVG_SCALE
        y *= SVG_SCALE
        fontsize *= SVG_SCALE // 10
        self.svg = SVG_TEXT.format(**locals())

class SvgDiagram:
    def add(self, component): 
        self.diagram.append(component.svg)

# Discussion:
# The way abstract factory works is to have a base abstraction-- DiagramFactory -- that will work
# for any factories we create. Then we create the concrete class SvgDiagramFactory for SVG
# implementations. Instead of using the same make_diagram(), we will define an SVG specific
# make_diagram() function (and the same for the rest, but left out in example). By naming the
# functions the same, create_diagram() can call the exact same lines for different factories.

# Issues: 1) Neither factories truly need a state of its own, so creating factory instances
# doesn't achieve much tbh 2) The code for SvgDiagramFactory and DiagramFactory are almost
# identical, a lot of duplication

# Notes: The default we use is instance method. We pass the instance "self" into the functions,
# and that contains the state of the instance. Here, we use the class method. We pass the
# uninstantiated class itself. It follows similar flow as static method, and is intended to be 
# a middleground. It gives us the encapsulation that static method has in terms of just taking
# inputs and giving outputs, but it still has access to the state.
# Another big advantage is that it is calling the class itself, and is aware of the class.
# So if there are two different classes Class1.make_diagram() and Class2.make_diagram(),
# the class method make_diagram() can properly inherit properties of each class.

# More Pythonic Approach
class DiagramFactory:
    @classmethod
    def make_diagram(Class, width, height):
        return Class.Diagram(width, height)
    
    @classmethod
    def make_rectangle(Class, x, y, width, height, fill='white', stroke='black'):
        return Class.Rectangle(x, y, width, height, fill, stroke)
    
    @classmethod
    def make_text(Class, x, y, text, fontsize=12):
        return Class.Text(x, y, text, fontsize)

    class Diagram:
        def __init__(self, width, height):
            self.width = width
            self.height = height
            self.diagram = DiagramFactory._create_rectangle(self.width,
                    self.height, DiagramFactory.BLANK)


        def add(self, component):
            for y, row in enumerate(component.rows):
                for x, char in enumerate(row):
                    self.diagram[y + component.y][x + component.x] = char


        def save(self, filenameOrFile):
            file = (None if isinstance(filenameOrFile, str) else
                    filenameOrFile)
            try:
                if file is None:
                    file = open(filenameOrFile, "w", encoding="utf-8")
                for row in self.diagram:
                    print("".join(row), file=file)
            finally:
                if isinstance(filenameOrFile, str) and file is not None:
                    file.close()
    
    # Won't add the rest....

class SvgDiagramFactory(DiagramFactory):
    class Diagram:
        def __init__(self, width, height):
            pxwidth = width * SvgDiagramFactory.SVG_SCALE
            pxheight = height * SvgDiagramFactory.SVG_SCALE
            self.diagram = [SvgDiagramFactory.SVG_START.format(**locals())]
            outline = SvgDiagramFactory.Rectangle(0, 0, width, height,
                    "lightgreen", "black")
            self.diagram.append(outline.svg)


        def add(self, component):
            self.diagram.append(component.svg)


        def save(self, filenameOrFile):
            file = (None if isinstance(filenameOrFile, str) else
                    filenameOrFile)
            try:
                if file is None:
                    file = open(filenameOrFile, "w", encoding="utf-8")
                file.write("\n".join(self.diagram))
                file.write("\n" + SvgDiagramFactory.SVG_END)
            finally:
                if isinstance(filenameOrFile, str) and file is not None:
                    file.close()

        # Won't add the rest....


# Discussion: We still need to create concrete classes for regular vs SVG. But now,
# instead of repeating methods like make_diagram(), we only define once in the base
# class DiagramFactory and inherit it into SVG (and future classes e.g. PNG).
# We also nested the Diagram, Text, Rectangle classes into the higher level
# factory classes for cleanliness. 

# Framework:
# This pattern is useful when you want to produce families of related objects 
# without specifying their concrete classes.

# E.g., you want to create a bunch of furnitures. Different chairs, tables,
# sofa. You also have styles like classic, modern, art deco. One solution
# is to create abstract factories. Base class FurnitureFactory() will 
# contain all the methods to generate furnitures e.g. coffee chairs, office chair, sofa. 
# Then subclasses ModernFurnitureFactory() will inherit the methods to create
# those furnitures, but also apply styling to them. There are only so many
# base types of furniture, and its not hard to extend new styling for all
# of them by just inheriting the base furniture factory.