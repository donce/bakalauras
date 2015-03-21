% clear; nk=30; load CR1; Chrom3Test3; % cia bus jau MOK ir TEST
% whos;

PLOT = [' k.b.y.m.r.c.g.k+b+y+m+r+c+g+koboyomorocogok*b*y*m*r*c*g*'];
D1=CR1(1:500,:);
D2=CR1(501:1000,:);
D3=CR1(4001:4500,:);

mok = [1:nk:500];
t = [1:500];
T = t;
t(mok) = 0;
Test = T(find(t > 0.5));
disp([size(mok), size(Test)]);

DM = [D1(mok, :); D2(mok, :); D3(mok, :)];
nmo = size(mok, 2);
p = 2;
disp(nmo);

M1 = mean(D1(mok, :));
M2 = mean(D2(mok, :));
M3 = mean(D3(mok, :));

Nsize = [nmo, nmo, nmo];
ro = 0.00001;

W_LDA = SarunoLDA_ro(DM, Nsize, ro);

d1 = D1(mok, :) * W_LDA;
d2 = D2(mok, :) * W_LDA;
d3 = D3(mok, :) * W_LDA;

m1 = M1 * W_LDA;
m2 = M2 * W_LDA;
m3 = M3 * W_LDA;

figure(4);
clf;
plot(d1(:, 1), d1(:, 2), 'k.', ...
     d2(:, 1), d2(:, 2), 'b.', ...
     d3(:, 1), d3(:, 2), 'r.');
hold on;
plot(m1(:, 1), m1(:, 2), 'ks', ...
     m2(:, 1), m2(:, 2), 'bs', ...
     m3(:, 1), m3(:, 2), 'rs', 'MarkerSize', 12);

for j = 1:nmo
  plot([m1(1), d1(j, 1)], [m1(2), d1(j, 2)], 'k-', ...
       [m2(1), d2(j, 1)], [m2(2), d2(j, 2)], 'b-', ...
       [m3(1), d3(j, 1)], [m3(2), d3(j, 2)], 'r-');
end;

d1 = D1(Test, :) * W_LDA;
d2 = D2(Test, :) * W_LDA;
d3 = D3(Test, :) * W_LDA;
DData = [d1; d2; d3];

figure(5);
clf;
hold off;
plot(d1(:, 1), d1(:, 2), 'k.', ...
     d2(:, 1), d2(:, 2), 'b.', ...
     d3(:, 1), d3(:, 2), 'm.');
hold on;

V(1) = min(DData(:, 1));
V(2) = max(DData(:, 1));
V(3) = min(DData(:, 2));
V(4) = max(DData(:, 2));

S1 = cov(d1);
S2 = cov(d2);
S3 = cov(d3);

mm1 = mean(d1);
mm2 = mean(d2);
mm3 = mean(d3);

WF = 2 * (mm1 - mm2) * inv(S1 + S2);
WF = [WF, -0.5 * WF * (mm1 + mm2)'];
plot_LDF([V(1), V(2), V(3), V(4)], WF, 'r-'); 

P12 = size([find(WF(1:p) * d1' + WF(p+1) < 0), ...
            find(WF(1:p) * d2' + WF(p+1) >= 0)], 2) / 1000; 
WF = 2 * (mm1 - mm3) * inv(S1+S3);
WF = [WF, -0.5 * WF * (mm1+mm3)'];
plot_LDF([V(1), V(2), V(3), V(4)], WF, 'r-'); 

P13 = size([find(WF(1:p) * d1' + WF(p+1) < 0), ...
            find(WF(1:p) * d3' + WF(p+1) >= 0)], 2) / 1000; 
WF = 2 * (mm2-mm3) * inv(S2+S3);
WF = [WF, -0.5 * WF * (mm2+mm3)'];
plot_LDF([V(1), V(2), V(3), V(4)], WF, 'r-'); 

P23 = size([find(WF(1:p) * d2' + WF(p+1) < 0), ...
            find(WF(1:p) * d3' + WF(p+1) >= 0)], 2) / 1000; 
disp([P12, P13, P23]);
