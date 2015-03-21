% W_LDA=SarunoLDA_ro(Data,Nsize,ro)
% Data -duomenys sum(Nsize) eiluciu, ir dim stulpeliu
%ro reguliarizacija: gali buti ro=0,0,001, 0.1 ir net  0.9 0.9999
% Nsize, kiek vektoriu kiekvienoje klaseje

function W_LDA = SarunoLDA_ro(Data, Nsize, ro)
Ends = cumsum(Nsize);
Nklas = size(Nsize, 2);
dim = size(Data, 2);
Data = double(Data);
N1size = [1, Nsize(1 : Nklas - 1)];
m = [];

Sw=zeros(dim, dim);
for j = 1 : Nklas
    m(j, 1 : dim) = mean(Data(N1size(j) : Ends(j), :));
    Sw = Sw + Nsize(j) * cov(Data(N1size(j) : Ends(j), :));
end;

Sw = Sw ./ Ends(end);
Sw = Sw * (1 - ro) + diag(diag(Sw) * ro);
Sb = cov(m);

%%%%%%%%%%%%%%%%%%%%%%%% Calculating Fisher discriminant basis's
% We want to maximise the Between Scatter Matrix, while minimising the
% Within Scatter Matrix. Thus, a cost function J is defined, so that this condition is satisfied.

[J_eig_vec, J_eig_val] = eig(Sw \ Sb);
[s,ind] = sort(diag(J_eig_val), 'descend');

%%%%%%%%%%%%%%%%%%%%%%%% Eliminating zero eigens and sorting in descend order
for i = 1 : Nklas - 1 
    W_LDA(:, i) = J_eig_vec(:, ind(i)); % Largest (C-1) eigen vectors of matrix J
end
