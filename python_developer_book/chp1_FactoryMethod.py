# Packages
import os
import sys
import tempfile

# -------------------------------------------------- #
# -------------------------------------------------- #
# Factory Method Pattern (Creational Pattern)

# This design pattern provides an interface for creating objects
# in a superclass, but allows subclasses to alter the type of objects
# created.

def main():
    #checkers = CheckersBoard() 
    #print(checkers)
    #chess = ChessBoard() 
    #print(chess)
    None

BLACK, WHITE = ("BLACK", "WHITE") 
class AbstractBoard:
    def __init__(self, rows, columns):
        self.board = [[None for _ in range(columns)] for _ in range(rows)] 
        self.populate_board()

    def populate_board(self):
        raise NotImplementedError()

    def __str__(self): 
        squares = []
        for y, row in enumerate(self.board): 
            for x, piece in enumerate(row):
                #square = console(piece, BLACK if (y + x) % 2 else WHITE)
                #squares.append(square)
            #squares.append("\n")
        
        #return #"".join(squares)

                return None