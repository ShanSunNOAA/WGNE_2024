wd1=1.4;wd2=2; w=0.44; hgt1=0.12; hgt2=0.08;dj=.02; 
wd1=1.4;wd2=2; w=0.42; hgt1=0.13; hgt2=0.10;dj=.03; 
xmin=2003;xmax=2019; bar_int=5; bar_int2=10;
sz_legend=8;
sz_xlbl=8;
sz_mk=5;
sz_tlt=16;

lat180 = (-89.5:89.5);
clat180 = cos(pi/180*lat180);
lat361 = (-90:.5:90);
clat361 = cos(pi/180*lat361);
i180=180;

start = [1 1 5]; % Start location along each coordinate
count = [360 180 1]; % Read until the end of each dimension

pth00 = ['/scratch1/BMC/gsd-fv3-dev/data_others/AOD/monthlymean/MODIS_']; %%%(i=1,j=1)=(-89.5, -179.5)
pth0 = ['/scratch1/BMC/gsd-fv3-dev/data_others/AOD/merra2/MERRA2_400.tavgM_2d_aer_Nx.']; %%% (-90,-89.5,-180,-179.375) 576x361
pth1 = ['/scratch2/BMC/gsd-hpcs/Shan.Sun/clim_togo_exp/COMROOT/clim0/gfs.'];
pth2 = ['/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/COMROOT/chm0/gfs.']; %%% -89.5,-88.5;0.5,1.5
y0=2002;

for mm=1:2
 if (mm==1) mon=5; amon='May';i1=0.04; j1=.91-hgt1-dj;end
 if (mm==2) mon=9; amon='Sept';i1=0.53; j1=.91-hgt1-dj;end
for number=0:3
  hgt=hgt2;
  if (number==0)
    hgt=hgt1;
  end
for yy=2003:2019
 flnm00= [pth00,num2str(yy,'%04.0f'),num2str(mon,'%02.0f'),'.nc']; 
 flnm0 = [pth0,num2str(yy,'%04.0f'),num2str(mon,'%02.0f'),'.nc4']; 
 flnm1 = [pth1,num2str(yy,'%04.0f'),num2str(mon,'%02i'),'01/00/var_wk_1x1.nc'];
 flnm2 = [pth2,num2str(yy,'%04.0f'),num2str(mon,'%02i'),'01/00/var_wk_1x1.nc'];

    f00=ncread(flnm00,'MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean');
    f0=ncread(flnm0,'TOTEXTTAU');
    f1=ncread(flnm1,'maod',start,count); 
    f2=ncread(flnm2,'maod',start,count); 

  if (number==0) 
    ymin=0.12; 
    ymax=0.22;
    znl00=nanmean(f00,1);
    gl00(yy-y0)=nansum(clat180.*znl00)/sum(clat180);

    znl0=mean(f0,1);
    gl0(yy-y0)=nansum(clat361.*znl0)/sum(clat361);

    znl1=mean(f1,1);
    gl1(yy-y0)=nansum(clat180.*znl1)/sum(clat180);

    znl2=mean(f2,1);
    gl2(yy-y0)=nansum(clat180.*znl2)/sum(clat180);
  end

  if (number > 0) % choose lat/lon
   if (number==1) %lon/lat: 0:30E, 0:30N;  Northern Africa
    ymin=0.2; 
    ymax=0.8;

    imax=360; jmax=180;
    ii11=1; ii21=ii11+imax/12;jj11=jmax/2; jj21=jj11+jmax/6;
    ii100=ii11+i180; ii200=ii100+imax/12; jj100=jj11; jj200=jj21;
    imax=576; jmax=360;
    ii10=imax/2;ii20=ii10+imax/12;jj10=jmax/2;jj20=jj10+jmax/6;
   end

   if (number==2) %lon/lat: 0:30E, 0:30S;  Southern Africa
    ymin=0.1; 
    ymax=0.6;

    imax=360; jmax=180;
    ii11=1; ii21=ii11+imax/12;jj21=jmax/2; jj11=jj21-jmax/6;
    ii100=ii11+i180; ii200=ii100+imax/12; jj200=jj21; jj100=jj11;
    imax=576; jmax=360;
    ii10=imax/2;ii20=ii10+imax/12;jj20=jmax/2;jj10=jj20-jmax/6;
   end

   if (number==3) %lon/lat: 100:130E, 15:45;  E Asian
    ymin=0.1; 
    ymax=0.6;

    imax=360; jmax=180;
    ii11=100; ii21=ii11+imax/12;jj11=jmax/2+15; jj21=jj11+jmax/6;
    ii100=ii11+i180; ii200=ii100+imax/12; jj100=jj11; jj200=jj21;
    imax=576; jmax=360;
    ii10=imax/2;ii20=ii10+imax/12;jj20=jmax/2;jj10=jj20-jmax/6;
   end

    p00=f00(ii100:ii200,:);
    pznl00=nanmean(p00,1);
    gl00(yy-y0)=nansum(clat180(jj100:jj200).*pznl00(jj100:jj200))/sum(clat180(jj100:jj200));

    p0=f0(ii10:ii20,:);
    pznl0=nanmean(p0,1);
    gl0(yy-y0)=nansum(clat361(jj10:jj20).*pznl0(jj10:jj20))/sum(clat361(jj10:jj20));

    p1=f1(ii11:ii21,:);
    pznl1=nanmean(p1,1);
    gl1(yy-y0)=nansum(clat180(jj11:jj21).*pznl1(jj11:jj21))/sum(clat180(jj11:jj21));

    p2=f2(ii11:ii21,:);
    pznl2=nanmean(p2,1);
    gl2(yy-y0)=nansum(clat180(jj11:jj21).*pznl2(jj11:jj21))/sum(clat180(jj11:jj21));
  end
end %yy

axes('position',[i1,j1,w,hgt]);
j1=j1-hgt2-dj;

xdim0=[2003:1:2019];
plot(xdim0,gl00,'k-o','linewidth',wd1,'MarkerSize',sz_mk); hold on
plot(xdim0,gl0,'g-o','linewidth',wd1,'MarkerSize',sz_mk); hold on
plot(xdim0,gl1,'b-o','linewidth',wd1,'MarkerSize',sz_mk); hold on;
plot(xdim0,gl2,'r-o','linewidth',wd1,'MarkerSize',sz_mk); hold on

xticks([2003 2007 2011 2015 2019])
set(gca,'xticklabel',[])
yy0=ymin-(ymax-ymin)*.05; dm=.5;
m=2003; text(m-dm,yy0,num2str(m,'%04.0f'),'FontSize',sz_xlbl)
m=2007; text(m-dm,yy0,num2str(m,'%04.0f'),'FontSize',sz_xlbl)
m=2011; text(m-dm,yy0,num2str(m,'%04.0f'),'FontSize',sz_xlbl)
m=2015; text(m-dm,yy0,num2str(m,'%04.0f'),'FontSize',sz_xlbl)
m=2019; text(m-dm,yy0,num2str(m,'%04.0f'),'FontSize',sz_xlbl)

if (number==0) title0=sprintf('%s %s %s',amon); out='fig4_znl_aod6_std.png';
 %text(2009,ymax*1.1,title0,'FontSize',9,'color','k')
  if (mm==1) mytlt=['(a) ',amon,' Global']; end
  if (mm==2) mytlt=['(b) ',amon,' Global']; end
end
if (number==1) title0=sprintf('%s %s %s','Zonal Mean AOD Components',amon,'2003-2019');
  if (mm==1) mytlt=['(c) ',amon,' Northern Africa']; end
  if (mm==2) mytlt=['(d) ',amon,' Northern Africa']; end
end
if (number==2) title0=sprintf('%s %s %s','Zonal Mean Dust AOD',amon,'2003-2019');
  if (mm==1) mytlt=['(e) ',amon,' Southern Africa']; end
  if (mm==2) mytlt=['(f) ',amon,' Southern Africa']; end
end
if (number==3) title0=sprintf('%s %s %s','Zonal Mean BC AOD',amon,'2003-2019');
  if (mm==1) mytlt=['(g) ',amon,' East Asia']; end
  if (mm==2) mytlt=['(h) ',amon,' East Asia']; end
end

title (mytlt,'Fontsize',sz_tlt,'Fontweight','normal');
%text(147.,.85*ymax,mytlt,'FontSize',sz_legend,'color','k')
%xticks([1:30:180])
%xticklabels({'','','','','',''})

if (number==0) 
  ave00=mean(gl00);
  ave0=mean(gl0);
  ave1=mean(gl1);
  ave2=mean(gl2) 
  ave = ['MODIS   (ave=',num2str(ave00,'%04.2f'),')'];
  text(xmax-5.5,.97*ymax,ave,'FontSize',sz_legend,'color','k')
  ave = ['MERRA2 (ave=',num2str(ave0,'%04.2f'),')'];
  text(xmax-5.5,.93*ymax,ave,'FontSize',sz_legend,'color','g')
  ave = ['ClimAer (ave=',num2str(ave1,'%04.2f'),')'];
  text(xmax-5.5,.89*ymax,ave,'FontSize',sz_legend,'color','b')
  ave = ['ProgAer (ave=',num2str(ave2,'%04.2f'),')'];
  text(xmax-5.5,.85*ymax,ave,'FontSize',sz_legend,'color','r')
end

%text(30.-ii0,yy0,'60^oS','FontSize',sz0)
%text(60.-ii0,yy0,'30^oS','FontSize',sz0)
%text(90.-ii0,yy0,'EQ','FontSize',sz0)
%text(120.-ii0,yy0,'30^oN','FontSize',sz0)
%text(150.-ii0,yy0,'60^oN','FontSize',sz0)

a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',sz_xlbl)
set(gca,'YTickLabelMode','auto')
grid
axis ([xmin xmax ymin ymax])
end %number
end %mon

axes('position',[0.35 .88 0.3 0.01])
title ('Global/Regional AOD','Fontsize',9,'Fontweight','normal');
axis off

orient tall
out='fig3_aod0_yr'
print(out,'-dpng')
print(out,'-depsc2')
