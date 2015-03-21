%  PLOTD Plot discriminant function
% Sarunas Raudys
% plot_LDF([V(1),V(2),V(3),V(4)],w,'k-') 
% srityje, apibreziamoje x1 ir x2-ju koordinatemis V, nupiesa skiriamaji pavirsiu
% w(1:2)*(x1,x2)'+w(3)=0.
% s tai kokia spalva ir kokia linija piesti % klases skiriamaja tiese

function plot_LDF(V, w, s)
	w1 = w(1:2);
  w0 = w(3);
	x = sort([V(1), V(2), (-w1(2) * V(3) - w0) / w1(1), (-w1(2) * V(4) - w0) / w1(1)]);
	x = x(2:3);
	y = (-w1(1) * x - w0) / w1(2);
	plot(x, y, s);
return
