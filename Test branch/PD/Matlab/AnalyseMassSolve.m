% In DiffPD, (M/h**2+\partial E(x))*(\partial x/\partial y)=M/h**2. 
% The difference of the value of (\partial x/\partial y) in normal mass
% distribution and weighted mass distribution by positional constraint.

M = lhs - Aq;

original_M = M;
fix_par = [1, 2, 5, 6];
for i=1:length(fix_par)
    idx = fix_par(i);
    original_M(idx, idx) = original_M(idx, idx) - 1.e9;
end
A_position = zeros(8,8);
for i=1:length(fix_par)
    idx = fix_par(i);
    A_position(idx, idx) = 1.e15;
end

dx_dy = (original_M+A_position+Aq+dA) \ original_M