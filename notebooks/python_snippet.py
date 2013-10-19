#!/usr/bin/env python

import numpy as np

A = np.random.random((4,4))
Ainv = np.linalg.inv(A)

print 'A ', A
print 'Ainv', Ainv
print 'A*Ainv', np.dot(A,Ainv)
