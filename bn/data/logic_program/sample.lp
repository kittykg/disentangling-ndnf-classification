VAR a 0 1
VAR b 0 1
VAR c 0 1


a(1,T) :- c(1,T-1).
b(1,T) :- a(1,T-1).
c(1,T) :- a(1,T-1), b(1,T-1).
