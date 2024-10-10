wd1=1.4;wd2=2; w=0.44; hgt1=0.14; hgt2=0.1;dh=.010; 
ymin=0; xmin=1;xmax=180; bar_int=5; bar_int2=10;
sz_legend=8;

lat180 = (-89.5:89.5);
clat180 = cos(pi/180*lat180);
lat361 = (-90:.5:90);
clat361 = cos(pi/180*lat361);

start = [1 1 5]; % Start location along each coordinate
count = [360 180 1]; % Read until the end of each dimension

pth00 = ['/scratch1/BMC/gsd-fv3-dev/data_others/AOD/monthlymean/MODIS_'];
pth0 = ['/scratch1/BMC/gsd-fv3-dev/data_others/AOD/merra2/MERRA2_400.tavgM_2d_aer_Nx.'];
pth2 = ['/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/COMROOT/chm0/gfs.'];
y0=2002;

for mm=1:2
 if (mm==1) mon=5; amon='May';i1=0.04; j1=0.80;end
 if (mm==2) mon=9; amon='Sep';i1=0.52; j1=0.80;end
for number=0:5
  hgt=hgt2;
  if (number==0)
    hgt=hgt1;
  end
for yy=2003:2018
 flnm00= [pth00,num2str(yy,'%04.0f'),num2str(mon,'%02.0f'),'.nc']; 
 flnm0 = [pth0,num2str(yy,'%04.0f'),num2str(mon,'%02.0f'),'.nc4']; 
 flnm2 = [pth2,num2str(yy,'%04.0f'),num2str(mon,'%02i'),'01/00/var_wk_1x1.nc'];

  if (number==0) 
    f00=ncread(flnm00,'MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean');
    f0=ncread(flnm0,'TOTEXTTAU');
    f2=ncread(flnm2,'maod',start,count); end
    znl00=mean(f00,1);
    aer00(:,yy-y0)=znl00;
  if (number==1) 
    f0=ncread(flnm0,'SUEXTTAU');
    f2=ncread(flnm2,'maodsulf',start,count); end
  if (number==2) 
    f0=ncread(flnm0,'DUEXTTAU');
    f2=ncread(flnm2,'maoddust',start,count); end
  if (number==3) 
    f0=ncread(flnm0,'BCEXTTAU');
    f2=ncread(flnm2,'maodbc',start,count); end
  if (number==4) 
    f0=ncread(flnm0,'OCEXTTAU');
    f2=ncread(flnm2,'maodoc',start,count); end
  if (number==5) 
    f0=ncread(flnm0,'SSEXTTAU');
    f2=ncread(flnm2,'maodseas',start,count); end

 znl0=mean(f0,1);
 znl2=mean(f2,1);

 aer0(:,yy-y0)=znl0;
 aer2(:,yy-y0)=znl2;
end %yy

std00(:)=nanstd(aer00,0,2);
ave00(:)=nanmean(aer00,2); 
xdim00=[1:1:length(ave00)];
ya_modis=sum(clat180.*ave00)/sum(clat180);

std0(:)=nanstd(aer0,0,2);
ave0(:)=nanmean(aer0,2); 
ya_merra2=sum(clat361.*ave0)/sum(clat361);

std2(:)=nanstd(aer2,0,2);
ave2(:)=nanmean(aer2,2);
ya_chm=sum(clat180.*ave2)/sum(clat180);

xdim0=[1:1:length(ave0)]/2;
xdim2=[1:1:length(ave2)];

axes('position',[i1,j1,w,hgt]);
j1=j1-hgt2-dh

plot(xdim0,ave0,'g','linewidth',wd1); hold on
plot(xdim2,ave2,'r','linewidth',wd1); hold on
e=errorbar(xdim0(1:bar_int2:360),ave0(1:bar_int2:360),std0(1:bar_int2:360),'g'); hold on
e.LineStyle = 'none';
e=errorbar(xdim2(1:bar_int:180),ave2(1:bar_int:180),std2(1:bar_int:180),'r'); hold on
e.LineStyle = 'none';

  if (number==0) 
    plot(xdim00,ave00,'k','linewidth',wd1); hold on
    e=errorbar(xdim00(1:bar_int:180),ave00(1:bar_int:180),std00(1:bar_int:180),'k'); hold on
    e.LineStyle = 'none';
   ;title0=sprintf('%s %s %s','Zonal Mean AOD',amon,'2003-2019'); out='fig4_znl_aod6_std.png';
    title0=sprintf('%s %s %s',amon); out='fig4_znl_aod6_std.png';
   %title(title0,'FontSize',10,'Fontweight','normal');
   %ax = gca;
   %ax.FontSize = 10;
    text(84.,0.54,title0,'FontSize',9,'color','k')
    ymax=.5;
    mylbl='   Total';
% be consistent by using the ave in the global plot
    ya_modis=0.17
    if (mm==1) ya_merra2=0.14; end
    if (mm==2) ya_merra2=0.15; end
  end
  if (number==1) title0=sprintf('%s %s %s','Zonal Mean AOD Components',amon,'2003-2019');out='znl_aod1_std.png';
    ymax=.15;
    mylbl=' Sulfate';
  end
  if (number==2) title0=sprintf('%s %s %s','Zonal Mean Dust AOD',amon,'2003-2019');out='znl_aod2_std.png';
    ymax=.25;
    mylbl='    Dust';
  end
  if (number==3) title0=sprintf('%s %s %s','Zonal Mean BC AOD',amon,'2003-2019');out='znl_aod3_std.png';
    ymax=.025;
    mylbl='      BC';
  end
  if (number==4) title0=sprintf('%s %s %s','Zonal Mean OC AOD',amon,'2003-2019');out='znl_aod4_std.png';
    ymax=.15;
    mylbl='      OC';
  end
  if (number==5) title0=sprintf('%s %s %s','Zonal Mean Sea Salt AOD',amon,'2003-2019');out='znl_aod5_std.png';
    ymax=.15;
    mylbl='Sea Salt';
  end

%text(7.,.85*ymax,mylbl,'FontSize',9,'color','k')
text(147.,.85*ymax,mylbl,'FontSize',sz_legend,'color','k')
xticks([1:30:180])
xticklabels({'','','','','',''})
if (number==0) 
  ave = ['MODIS   (ave=',num2str(ya_modis,'%04.2f'),')'];
  text(7.,.92*ymax,ave,'FontSize',sz_legend,'color','k')
  ave = ['MERRA2 (ave=',num2str(ya_merra2+0.01,'%04.2f'),')'];
  text(7.,.81*ymax,ave,'FontSize',sz_legend,'color','g')
  ave = ['ProgAer (ave=',num2str(ya_chm,'%04.2f'),')'];
  text(7.,.70*ymax,ave,'FontSize',sz_legend,'color','r')
else
  ave = ['MERRA2 (ave=',num2str(ya_merra2,'%05.3f'),')'];
  text(7.,.91*ymax,ave,'FontSize',sz_legend,'color','g')
  ave = ['ProgAer (ave=',num2str(ya_chm,'%05.3f'),')'];
  text(7.,.79*ymax,ave,'FontSize',sz_legend,'color','r')
end

grid
if (number==5) 
ii0=5.;
yy0=-(ymax-ymin)*.1;
sz0=7;
text(30.-ii0,yy0,'60^oS','FontSize',sz0)
text(60.-ii0,yy0,'30^oS','FontSize',sz0)
text(90.-ii0,yy0,'EQ','FontSize',sz0)
text(120.-ii0,yy0,'30^oN','FontSize',sz0)
text(150.-ii0,yy0,'60^oN','FontSize',sz0)
end
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',6)
set(gca,'YTickLabelMode','auto')

%ave = ['Timely\_aer (ave=',num2str(aer2,'%04.2f'),')'];

%if (number==5) 
%xticklabels({'','60^oS','30^oS','EQ','30^oN','60^oN'})
%end
%yticks([0:0.1:0.3])
%yticklabels({'0','0.1','0.2','0.3'})

axis ([xmin xmax ymin ymax])
end %number
end %mon

axes('position',[0.35 .94 0.3 0.01])
title ('Zonal Mean AOD 2003-2019','Fontsize',10,'Fontweight','normal');
axis off

orient tall
%print -depsc2 fig5_znl_aod5_std_1run_merra2.eps
%print -djpeg  fig5_znl_aod5_std_1run_merra2.jpg
out='fig4_znl_aod6_std_2obs'
print(out,'-dpng')
print(out,'-depsc2')
