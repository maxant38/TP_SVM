function gradH=gradH(alpha,A,u)
%  fonction renvoyant la valeur du gradient du lagrangien H
gradH = - A*alpha + u ;
end

