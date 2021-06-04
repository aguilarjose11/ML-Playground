Jose's Documentation Format
===========================

Welcome! Here I present yet another Python documentation format for python 3.\*+. This documentation "Manifesto" hereby presents documentation format that attempts to maintain simplicity and robustness in mind.

This project is in development and is to not be used officially yet.

- \{\} Defines a must-use

Definition table
----------------

- [a]()
- [b]()
- [c]()


Definitions
-----------

### File/Package Documentation

Package documentation must start and end with double-colon comments: """ --- """. The format of the documentation is:
``` python
""" {Short one line description of file}

{Detailed description of file}

global variables
---------------

variable -> type
- {Short description}

functions
----------

function -> return

* {Short description}

param

- parameter -> type
  - {One line description}

classes
-------

Class name

__init__
* functions

```
