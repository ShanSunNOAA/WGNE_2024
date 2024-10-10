wd2=1; w=0.91; hgt=0.32; hspace=0.03; vspace=0.10; i1=0.07; dj=.10; sz8=8;
j1=.96;
lat = linspace(-89.5, 89.5, 180); 
w1 = cosd(lat');

truth=1; %modis
truth=2; %merra2
mem2=4; % 5th is clim0
lat2=180; lon2=360; 

var_modis='MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean'; 
var_merra2='TOTEXTTAU'; 

for ncase=1:2
 j1=j1-hgt-dj
 if (ncase==1)
    flnm1 = ['/scratch1/BMC/gsd-fv3-dev/data_others/AOD/AOD_MODIS_2003_2019_1deg_daily/MODIS_AOD_May2003_2019.nc'];
    flnm2= ['/scratch1/BMC/gsd-fv3-dev/data_others/AOD/AOD_MERRA2_2003_2019_1deg_daily/merra2_aod_may2003_2019.nc'];
    flnm{1}= '/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm1/aer_daily_may_2003_2019.nc';
    flnm{2}= '/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm2/aer_daily_may_2003_2019.nc';
    flnm{3}= '/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm4/aer_daily_may_2003_2019.nc';
    flnm{4}= '/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm5/aer_daily_may_2003_2019.nc';
    flnm0= '/scratch1/BMC/wrfruc/Shan.Sun/wgne_clim/clim0/aer_daily_may_2003_2019.nc';
    t2=527;
 end
 if (ncase==2)
    flnm1 = ['/scratch1/BMC/gsd-fv3-dev/data_others/AOD/AOD_MODIS_2003_2019_1deg_daily/MODIS_AOD_Sep2003_2019.nc'];
    flnm2= ['/scratch1/BMC/gsd-fv3-dev/data_others/AOD/AOD_MERRA2_2003_2019_1deg_daily/merra2_aod_sep2003_2019.nc'];
    flnm{1}= '/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm1/aer_daily_sep_2003_2019.nc';
    flnm{2}= '/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm2/aer_daily_sep_2003_2019.nc';
    flnm{3}= '/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm4/aer_daily_sep_2003_2019.nc';
    flnm{4}= '/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm5/aer_daily_sep_2003_2019.nc';
    flnm0= '/scratch1/BMC/wrfruc/Shan.Sun/wgne_clim/clim0/aer_daily_sep_2003_2019.nc';
    t2=510;
 end
%ncdisp(flnm1)

  monlg=t2/17;
  mon0=monlg*.5;
  start = [1 1 1]; count = [lon2 lat2 t2];
  work = ncread(flnm1,var_modis,start,count);
  znlmo=nanmean(work,1);
  for n=1:t2
  validIndices = ~isnan(znlmo(1,:,n));
  zclean=znlmo(1,validIndices,n);
  wclean=w1(validIndices);
  obsmo(n) = sum(zclean * wclean) / sum(wclean); 
  end
  
  work = ncread(flnm2,var_merra2,start,count);
  znl_m=mean(work,1);
  for n=1:t2
  obsme(n) = sum(znl_m(1,:,n) * w1) / sum(w1); 
  end
  
  work = ncread(flnm0,'aod550',start,count);
  znlm=mean(work,1);
  for n=1:t2
  vclim(n) = sum(znlm(1,:,n) * w1) / sum(w1); 
  end
  
  for mem=1:mem2
    work = ncread(flnm{mem},'aod550',start,count);
    znl_m=mean(work,1);
    for n=1:t2
      var1d(mem,n) = sum(znl_m(1,:,n) * w1) / sum(w1);
    end
  end
  var1dm=mean(var1d,1);
  
  for n=1:t2
  vmax(n)=max(var1d(:,n));
  vmin(n)=min(var1d(:,n));
  end
  xdim = [1:1:t2];
  x_fill = [xdim, fliplr(xdim)];              % X values for shading (top curve first)
  y_fill = [vmin, fliplr(vmax)];            % Corresponding Y values (flip for bottom curve)
  
  ymin=0.12;ymax=0.24;
  ave00=mean(obsmo);
  ave0=mean(obsme);
  if (ncase==1) ave0=ave0+0.01; end  % to match with Fig.1
  avep=mean(var1dm);
  avec=mean(vclim);
  vclim(1,monlg:monlg:t2)=NaN;
   vmin(1,monlg:monlg:t2)=NaN;
   vmax(1,monlg:monlg:t2)=NaN;
   obsmo(1,monlg:monlg:t2)=NaN;
   obsme(1,monlg:monlg:t2)=NaN;
  axes('position',[i1 j1 w hgt])
  plot(xdim,obsmo,'k',xdim,obsme,'g',xdim,vclim,'b','linewidth',wd2); hold on
  plot(xdim,vmin,'r',xdim,vmax,'r'); hold on
  fill(x_fill, y_fill, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
  grid
  xticks([mon0:monlg:t2])
  xticklabels({'2003','','2005','','2007','','2009','','2011','','2013','','2015','','2017','','2019'});
  ytickformat('%.2f')
  
  ave = ['MODIS   (ave=',num2str(ave00,'%04.2f'),')'];
  text(t2*.7,.97*ymax,ave,'FontSize',sz8,'color','k')
  ave = ['MERRA2 (ave=',num2str(ave0,'%04.2f'),')'];
  text(t2*.7,.93*ymax,ave,'FontSize',sz8,'color','g')
  ave = ['ProgAer (ave=',num2str(avep,'%04.2f'),')'];
  text(t2*.7,.89*ymax,ave,'FontSize',sz8,'color','r')
  ave = ['ClimAer (ave=',num2str(avec,'%04.2f'),')'];
  text(t2*.7,.85*ymax,ave,'FontSize',sz8,'color','b')

  axis ([1 t2 ymin ymax])

  if (ncase==1)
    text(t2*.07,.95*ymax,'May 2003-2019','FontSize',9,'color','k')
    axes('position',[0.38 0.86 0.3 0.01])
    title ('Global Mean AOD','Fontsize',10,'Fontweight','normal');
    axis off
  end
  if (ncase==2)
    text(t2*.07,.95*ymax,'Sept 2003-2019','FontSize',9,'color','k')
  end
  clear obsmo obsme vclim vmin vmax
end

out='fig1_aod_gl_4lines'
print(out,'-depsc2')
print(out,'-dpng')
