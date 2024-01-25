clear
clf

wd1=1.5; wd2=2; w=.40; dw=.05; hgt=.16; dh=.04; i0=0.10; j0=0.93;

%x1=    categorical({'NHX','SHX','N Afr','E Asian','S Afr','S Ame'});
%x1= reordercats(x1,{'NHX','SHX','N Afr','E Asian','S Afr','S Ame'});
x1=    categorical({'Wk1','Wk2','Wk3','Wk4','Wk3-4'});
x1= reordercats(x1,{'Wk1','Wk2','Wk3','Wk4','Wk3-4'});

mon=5; 
exp=["noa0","clim","chm"];
var=["precip","t2m","precip"];

flnm = ['/scratch2/BMC/gsd-fv3-dev/sun/post_p8_f90/acc_3var/acc0_noa0_t2m_05.txt'];
fid=fopen(flnm,'rt');
x01=textscan(fid,'%f %f %f %f %f');
t01=cell2mat(x01);
fclose(fid);

flnm = ['/scratch2/BMC/gsd-fv3-dev/sun/post_p8_f90/acc_3var/acc0_clim_t2m_05.txt'];
fid=fopen(flnm,'rt');
x01=textscan(fid,'%f %f %f %f %f');
t02=cell2mat(x01);
fclose(fid);

flnm = ['/scratch2/BMC/gsd-fv3-dev/sun/post_p8_f90/acc_3var/acc0_chm_t2m_05.txt'];
fid=fopen(flnm,'rt');
x01=textscan(fid,'%f %f %f %f %f');
t03=cell2mat(x01);
fclose(fid);

flnm = ['/scratch2/BMC/gsd-fv3-dev/sun/post_p8_f90/acc_3var/acc0_noa0_precip_05.txt'];
fid=fopen(flnm,'rt');
x01=textscan(fid,'%f %f %f %f %f');
p01=cell2mat(x01);
fclose(fid);

flnm = ['/scratch2/BMC/gsd-fv3-dev/sun/post_p8_f90/acc_3var/acc0_clim_precip_05.txt'];
fid=fopen(flnm,'rt');
x01=textscan(fid,'%f %f %f %f %f');
p02=cell2mat(x01);
fclose(fid);

flnm = ['/scratch2/BMC/gsd-fv3-dev/sun/post_p8_f90/acc_3var/acc0_chm_precip_05.txt'];
fid=fopen(flnm,'rt');
x01=textscan(fid,'%f %f %f %f %f');
p03=cell2mat(x01);
fclose(fid);

flnm = ['/scratch2/BMC/gsd-fv3-dev/sun/post_p8_f90/acc_3var/acc0_noa0_h500_05.txt'];
fid=fopen(flnm,'rt');
x01=textscan(fid,'%f %f %f %f %f');
h01=cell2mat(x01);
fclose(fid);

flnm = ['/scratch2/BMC/gsd-fv3-dev/sun/post_p8_f90/acc_3var/acc0_clim_h500_05.txt'];
fid=fopen(flnm,'rt');
x01=textscan(fid,'%f %f %f %f %f');
h02=cell2mat(x01);
fclose(fid);

flnm = ['/scratch2/BMC/gsd-fv3-dev/sun/post_p8_f90/acc_3var/acc0_chm_h500_05.txt'];
fid=fopen(flnm,'rt');
x01=textscan(fid,'%f %f %f %f %f');
h03=cell2mat(x01);
fclose(fid);

fontl=9;
for nplot=1:6

if (nplot==1) i1=i0; j1=j0-hgt;        titl0='NHX T2m'; end
if (nplot==2) i1=i0+w+dw;              titl0='SHX T2m'; end
if (nplot==3) i1=i0; j1=j0-hgt*2-dh;   titl0='NHX Precip'; end
if (nplot==4) i1=i0+w+dw;              titl0='SHX Precip'; end
if (nplot==5) i1=i0; j1=j0-hgt*3-dh*2; titl0='NHX H500'; end
if (nplot==6) i1=i0+w+dw;              titl0='SHX H500'; end
axes('position',[i1,j1,w,hgt])
if (nplot<=2) 
y(1:5,1)=t01(nplot,1:5);
y(1:5,2)=t02(nplot,1:5);
y(1:5,3)=t03(nplot,1:5); 
elseif  (nplot==3) || (nplot==4)  
y(1:5,1)=p01(nplot-2,1:5);
y(1:5,2)=p02(nplot-2,1:5);
y(1:5,3)=p03(nplot-2,1:5); 
elseif  (nplot==5) || (nplot==6)  
y(1:5,1)=h01(nplot-4,1:5);
y(1:5,2)=h02(nplot-4,1:5);
y(1:5,3)=h03(nplot-4,1:5); end

 b=bar(x1,y);
 ylow=.0; yhigh=1.;
 ylim([ylow yhigh]);
 b(1).FaceColor = 'c';
 b(2).FaceColor = 'b';
 b(3).FaceColor = 'r';
 if (nplot < 5) set(gca,'xticklabel',[]); end
 if (nplot == 1) legend('NoAer','ClimAer','ProgAer'); end
 grid
 title (titl0,'Fontsize',fontl,'Fontweight','normal');

end

axes('position',[0.35 j0+.01 0.3 0.01])
title ('(a) ACC May 2003-2019','Fontsize',9,'Fontweight','normal');
axis off

print -djpeg  fig11a_bar_nhx_shx_may.jpg
print -depsc2 fig11a_bar_nhx_shx_may.eps
