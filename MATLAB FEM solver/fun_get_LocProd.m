function [Prod] = fun_get_LocProd(Grad)
% Computes explicitly Prod = Grad.' * Grad where Grad is a tensor of nb_t 
% slices of size 2 x 3, reshaped in a 6 x nb_t table.


Prod(1,:) = Grad(1,:).^2 + Grad(2,:).^2;
Prod(2,:) = Grad(3,:) .* Grad(1,:) + Grad(4,:) .* Grad(2,:);
Prod(3,:) = Grad(5,:) .* Grad(1,:) + Grad(6,:) .* Grad(2,:);
Prod(4,:) = Prod(2,:);
Prod(5,:) = Grad(3,:).^2 + Grad(4,:).^2;
Prod(6,:) = Grad(5,:) .* Grad(3,:) + Grad(6,:) .* Grad(4,:);
Prod(7,:) = Prod(3,:);
Prod(8,:) = Prod(6,:);
Prod(9,:) = Grad(5,:).^2 + Grad(6,:).^2;