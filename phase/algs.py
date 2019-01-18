"""
This is the main interface to the applications.

The basic idea is that, the use can create a object to do phase retrieval.
The reason that I would like to use class is that after a short discussion with Zhen,
I realized that class makes it easier for the user to use. Also, in the algorithm, to
accelerate the calculation, I need to initialize a lot of variables before the
calculation. Therefore, it would be very clumsy to use dictionary or the other
methods for the users to use.
"""


class BaseCDI:
    pass
