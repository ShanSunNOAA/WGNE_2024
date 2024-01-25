load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin

  yr1=2003
  yr2=2003
  yr1=9999
  yr2=9999
  mo1=5
  mo2=9
  gl=1
  bias=0

; only 1 can be true
  do_aod=0
  cld=0 
  toa=1
  net=0
  sw=0
  albt=0
  modis_aod_ave=new(17,float)

  pth_chm = "/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/COMROOT/chm0/"
  pth_clim ="/scratch2/BMC/gsd-hpcs/Shan.Sun/clim_togo_exp/COMROOT/clim0/"
  pth_ctrl0="/scratch1/BMC/wrfruc/Shan.Sun/wfw_wgne_ctrl_exp/COMROOT/noa00/"
  ncrec0=0 ; monthly
  ncrec4=4 ; monthly

  if (gl.eq.1) then
   lat1=0;
   lat2=179;
   lon1=0;
   lon2=359;
  else
   lat1=60;
   lat2=120;
   lon1=0;
   lon2=60;
  end if

  ymin=0
  ymax=1
  first=1
  rad    = 4.0*atan(1.0)/180.0
  nn=0
  do mon=mo1,mo2,mo2-mo1
    if (mon.eq.5) then
      amon="May"
    end if
    if (mon.eq.9) then
      amon="Sep"
    end if
    patha=pth_chm
    pathb=pth_clim
    pathc=pth_ctrl0
  do yr=yr1,yr2
    if (yr.eq.9999) then
      modis="MODIS_"+amon+"_2003_2019_ave.nc"
      ceres="CERES_2003_2019_"+amon+"_ave.nc"
    else
      modis="MODIS_"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".nc"
      ceres="CERES_"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".nc"
    end if
    fpng= "map_znl6_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    sfcdata1= "gfs."+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+"01/00/var_wk_1x1.nc"
    fin1= addfile(patha+sfcdata1,"r");
    fin2= addfile(pathb+sfcdata1,"r");
    fin3= addfile(pathc+sfcdata1,"r");
    lonn=fin1->lon(lon1:lon2);
    latn=fin1->lat(lat1:lat2);
    clat   = cos(latn*rad)

    fout0= "htfx_gl_ceres_"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".txt"
    fout1= "htfx_gl_chem"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".txt"
    fout2= "htfx_gl_gf_no_aer"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".txt"
    fout3= "htfx_gl_gf_aer"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".txt"

    if (do_aod.eq.1 ) then
     var=" AOD "
     obs=" MODIS "
    end if

    path0 = "/scratch1/BMC/gsd-fv3-dev/data_others/AOD/monthlymean/"
    fin0= addfile(path0+modis,"r");
    print(fin0)
    fpng= "aod_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    if (gl.eq.1) then
      x90=fin0->MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean(lat1:lat2,lon1:lon2); monthly
      modis_file="/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/glb_aod/aod_gl_modis_2003_2019_"+sprinti("%0.2i",mon)+"_sht.txt"
    else
      x90=fin0->MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean(lat1:lat2,lon1+180:lon2+180); monthly
      modis_file="/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/glb_aod/aod_gl_modis_2003_2019_"+sprinti("%0.2i",mon)+"_sht.txt"
    end if
    modis_aod_ave=asciiread(modis_file,(/17/),"float")
    x91=fin1->maod(ncrec4,lat1:lat2,lon1:lon2); monthly
    x92=fin2->maod(ncrec4,lat1:lat2,lon1:lon2); monthly
    print(fin3)
    x93=fin3->AOD_550(ncrec4,lat1:lat2,lon1:lon2); monthly

    x90@_FillValue = -9999

    ave_x0 = wgt_areaave(x90, clat, 1.0, 1)
    ave_x1 = wgt_areaave(x91, clat, 1.0, 1)
    ave_x2 = wgt_areaave(x92, clat, 1.0, 1)
    ave_x3 = wgt_areaave(x93, clat, 1.0, 1)
    print(modis_aod_ave)
    if (yr.lt.9999) then
      ave_x0=modis_aod_ave(yr-2003)
    else
      ave_x0=avg(modis_aod_ave)
    end if
    znl_aod=new((/2,dimsizes(x91&lat)/),float)
   ;znl_aod(0,:)=dim_avg_n_Wrap(x90,1)
    znl_aod(0,:)=-dim_avg_n_Wrap(x91,1)
    znl_aod(1,:)=-dim_avg_n_Wrap(x92,1)
   ;znl_aod(3,:)=dim_avg_n_Wrap(x93,1)

    path0 = "/scratch1/BMC/gsd-fv3-dev/data_others/CERES_data/"
    fin0= addfile(path0+ceres,"r");
    print(fin0)
   if (bias.eq.0) then
    if (toa.eq.1) then
      fpng= "toa_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    end if
    if (net.eq.1) then
      fpng= "netsfc_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    end if
    if (sw.eq.1) then
      fpng= "sw_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    end if
    if (cld.eq.1) then
      fpng= "cld_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    end if
   else
    if (toa.eq.1) then
      fpng= "dtoa_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    end if
    if (net.eq.1) then
      fpng= "dnetsfc_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    end if
    if (sw.eq.1) then
      fpng= "dsw_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    end if
    if (cld.eq.1) then
      fpng= "dcld_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    end if
   end if; base

    x10=fin0->solar_mon(ncrec0,lat1:lat2,lon1:lon2); monthly
    x20=fin0->toa_sw_all_mon(ncrec0,lat1:lat2,lon1:lon2); monthly
    x30=fin0->toa_lw_all_mon(ncrec0,lat1:lat2,lon1:lon2); monthly
    x40=fin0->sfc_sw_down_all_mon(ncrec0,lat1:lat2,lon1:lon2); monthly
    x50=fin0->sfc_sw_up_all_mon(ncrec0,lat1:lat2,lon1:lon2); monthly
    x60=fin0->sfc_lw_down_all_mon(ncrec0,lat1:lat2,lon1:lon2); monthly
    x70=fin0->sfc_lw_up_all_mon(ncrec0,lat1:lat2,lon1:lon2); monthly
    x80=fin0->cldarea_total_daynight_mon(ncrec0,lat1:lat2,lon1:lon2); monthly
    x90=fin0->cldarea_total_daynight_mon(ncrec0,lat1:lat2,lon1:lon2); monthly

    x11=fin1->dswrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
    x21=fin1->uswrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
    x31=fin1->ulwrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
    x41=fin1->dswrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x51=fin1->uswrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x61=fin1->dlwrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x71=fin1->ulwrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x81=fin1->tcdc_aveclm(ncrec4,lat1:lat2,lon1:lon2); monthly
    x91=fin1->tcdc_aveclm(ncrec4,lat1:lat2,lon1:lon2); monthly
    y11=fin1->prate_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day
    y21=fin1->cprat_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day

    x12=fin2->dswrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
    x22=fin2->uswrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
    x32=fin2->ulwrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
    x42=fin2->dswrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x52=fin2->uswrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x62=fin2->dlwrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x72=fin2->ulwrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x82=fin2->tcdc_aveclm(ncrec4,lat1:lat2,lon1:lon2); monthly
    x92=fin2->tcdc_aveclm(ncrec4,lat1:lat2,lon1:lon2); monthly
    y12=fin2->prate_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day
    y22=fin2->cprat_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day

    x13=fin3->dswrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
    x23=fin3->uswrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
    x33=fin3->ulwrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
    x43=fin3->dswrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x53=fin3->uswrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x63=fin3->dlwrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x73=fin3->ulwrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    x83=fin3->tcdc_aveclm(ncrec4,lat1:lat2,lon1:lon2); monthly
    x93=fin3->tcdc_aveclm(ncrec4,lat1:lat2,lon1:lon2); monthly
    y13=fin3->prate_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day
    y23=fin3->cprat_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day
    albt0=fin3->cprat_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day
    albt1=fin3->cprat_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day
    albt2=fin3->cprat_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day
    albt3=fin3->cprat_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day
    albs0=fin3->cprat_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day
    albs1=fin3->cprat_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day
    albs2=fin3->cprat_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day
    albs3=fin3->cprat_ave(ncrec4,lat1:lat2,lon1:lon2)*86400; mm/day

    znl_toa=new((/4,dimsizes(x90&lat)/),float)
    znl_dtoa=new((/2,dimsizes(x90&lat)/),float)
    znl_net=new((/4,dimsizes(x90&lat)/),float)
    znl_sw =new((/4,dimsizes(x90&lat)/),float)
    znl_cld=new((/4,dimsizes(x90&lat)/),float)
    znl_albt=new((/4,dimsizes(x90&lat)/),float)
    znl_albs=new((/4,dimsizes(x90&lat)/),float)

    eps=0.
    eps=.1

    x10 = where(x10.gt.eps, x10, eps)
    x11 = where(x11.gt.eps, x11, eps)
    x12 = where(x12.gt.eps, x12, eps)
    x13 = where(x13.gt.eps, x13, eps)
    albt0=x20/x10
    albt1=x21/x11
    albt2=x22/x12
    albt3=x23/x13

   if (albt.eq.1) then
    x90 = albt0
    x91 = albt1
    x92 = albt2
    x93 = albt3
    var=" alb TOA "
    obs=" CERES "
   end if

   if (toa.eq.1) then
    x90=x10-x20-x30
    x91=x11-x21-x31
    x92=x12-x22-x32
    x93=x13-x23-x33
    var=" RF TOA "
    obs=" CERES "
   end if
   znl_toa(0,:)=dim_avg_n_Wrap(x10-x20-x30,1)
   znl_toa(1,:)=dim_avg_n_Wrap(x11-x21-x31,1)
   znl_toa(2,:)=dim_avg_n_Wrap(x12-x22-x32,1)
   znl_toa(3,:)=dim_avg_n_Wrap(x13-x23-x33,1)

   znl_dtoa(0,:)=znl_toa(1,:)-znl_toa(3,:)
   znl_dtoa(1,:)=znl_toa(2,:)-znl_toa(3,:)

   znl_albt(0,:)=dim_avg_n_Wrap(albt0,1)
   znl_albt(1,:)=dim_avg_n_Wrap(albt1,1)
   znl_albt(2,:)=dim_avg_n_Wrap(albt2,1)
   znl_albt(3,:)=dim_avg_n_Wrap(albt3,1)

   znl_albs(0,:)=dim_avg_n_Wrap(albs0,1)
   znl_albs(1,:)=dim_avg_n_Wrap(albs1,1)
   znl_albs(2,:)=dim_avg_n_Wrap(albs2,1)
   znl_albs(3,:)=dim_avg_n_Wrap(albs3,1)

   if (net.eq.1) then
    x90=x40-x50+x60-x70
    x91=x41-x51+x61-x71
    x92=x42-x52+x62-x72
    x93=x43-x53+x63-x73
    var=" Sfc Rad "
    obs=" CERES "
   end if
   znl_net(0,:)=dim_avg_n_Wrap(x40-x50+x60-x70,1)
   znl_net(1,:)=dim_avg_n_Wrap(x41-x51+x61-x71,1)
   znl_net(2,:)=dim_avg_n_Wrap(x42-x52+x62-x72,1)
   znl_net(3,:)=dim_avg_n_Wrap(x43-x53+x63-x73,1)

   if (sw.eq.1) then
    x90=x40
    x91=x41
    x92=x42
    x93=x43
    var=" Sfc SW Dwn "
    obs=" CERES "
   end if
   znl_sw(0,:)=dim_avg_n_Wrap(x40,1)
   znl_sw(1,:)=dim_avg_n_Wrap(x41,1)
   znl_sw(2,:)=dim_avg_n_Wrap(x42,1)
   znl_sw(3,:)=dim_avg_n_Wrap(x43,1)

   if (cld.eq.1) then
    x90=x80
    x91=x81
    x92=x82
    x93=x83
    var=" Cloud Cover (%) "
    obs=" CERES "
   end if
   znl_cld(0,:)=dim_avg_n_Wrap(x80,1)
   znl_cld(1,:)=dim_avg_n_Wrap(x81,1)
   znl_cld(2,:)=dim_avg_n_Wrap(x82,1)
   znl_cld(3,:)=dim_avg_n_Wrap(x83,1)

   ave_x0 = wgt_areaave(x90, clat, 1.0, 1)
   ave_x1 = wgt_areaave(x91, clat, 1.0, 1)
   ave_x2 = wgt_areaave(x92, clat, 1.0, 1)
   ave_x3 = wgt_areaave(x93, clat, 1.0, 1)

   ave_x10 = wgt_areaave(x10, clat, 1.0, 1)
   ave_x20 = wgt_areaave(x20, clat, 1.0, 1)
   ave_x30 = wgt_areaave(x30, clat, 1.0, 1)
   ave_x40 = wgt_areaave(x40, clat, 1.0, 1)
   ave_x50 = wgt_areaave(x50, clat, 1.0, 1)
   ave_x60 = wgt_areaave(x60, clat, 1.0, 1)
   ave_x70 = wgt_areaave(x70, clat, 1.0, 1)
   ave_x80 = wgt_areaave(x80, clat, 1.0, 1)
   write_table(fout0,"w",[/ave_x10,ave_x20,ave_x30,ave_x40,ave_x50,ave_x60,ave_x70,ave_x80/], "%8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f")

   ave_x11 = wgt_areaave(x11, clat, 1.0, 1)
   ave_x21 = wgt_areaave(x21, clat, 1.0, 1)
   ave_x31 = wgt_areaave(x31, clat, 1.0, 1)
   ave_x41 = wgt_areaave(x41, clat, 1.0, 1)
   ave_x51 = wgt_areaave(x51, clat, 1.0, 1)
   ave_x61 = wgt_areaave(x61, clat, 1.0, 1)
   ave_x71 = wgt_areaave(x71, clat, 1.0, 1)
   ave_x81 = wgt_areaave(x81, clat, 1.0, 1)
   ave_y11 = wgt_areaave(y11, clat, 1.0, 1)
   ave_y21 = wgt_areaave(y21, clat, 1.0, 1)
   write_table(fout1,"w",[/ave_x11,ave_x21,ave_x31,ave_x41,ave_x51,ave_x61,ave_x71,ave_x81,ave_y11,ave_y21/], "%8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f")

   ave_x12 = wgt_areaave(x12, clat, 1.0, 1)
   ave_x22 = wgt_areaave(x22, clat, 1.0, 1)
   ave_x32 = wgt_areaave(x32, clat, 1.0, 1)
   ave_x42 = wgt_areaave(x42, clat, 1.0, 1)
   ave_x52 = wgt_areaave(x52, clat, 1.0, 1)
   ave_x62 = wgt_areaave(x62, clat, 1.0, 1)
   ave_x72 = wgt_areaave(x72, clat, 1.0, 1)
   ave_x82 = wgt_areaave(x82, clat, 1.0, 1)
   ave_y12 = wgt_areaave(y12, clat, 1.0, 1)
   ave_y22 = wgt_areaave(y22, clat, 1.0, 1)
   write_table(fout2,"w",[/ave_x12,ave_x22,ave_x32,ave_x42,ave_x52,ave_x62,ave_x72,ave_x82,ave_y12,ave_y22/], "%8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f")

   ave_x13 = wgt_areaave(x13, clat, 1.0, 1)
   ave_x23 = wgt_areaave(x23, clat, 1.0, 1)
   ave_x33 = wgt_areaave(x33, clat, 1.0, 1)
   ave_x43 = wgt_areaave(x43, clat, 1.0, 1)
   ave_x53 = wgt_areaave(x53, clat, 1.0, 1)
   ave_x63 = wgt_areaave(x63, clat, 1.0, 1)
   ave_x73 = wgt_areaave(x73, clat, 1.0, 1)
   ave_x83 = wgt_areaave(x83, clat, 1.0, 1)
   ave_y13 = wgt_areaave(y13, clat, 1.0, 1)
   ave_y23 = wgt_areaave(y23, clat, 1.0, 1)
   write_table(fout3,"w",[/ave_x13,ave_x23,ave_x33,ave_x43,ave_x53,ave_x63,ave_x73,ave_x83,ave_y13,ave_y23/], "%8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f")
   print("ave_x0="+ave_x0+" ave_x1="+ave_x1+" ave_x2="+ave_x2+" ave_x3="+ave_x3)

  if (first.eq.1) then
   ;wks = gsn_open_wks("x11", "contour_map")   ;-- open a workstation
    wks = gsn_open_wks("pdf",get_script_prefix_name())
   ;wks = gsn_open_wks("png",fpng)
    plot = new(2,graphic) ; create a plot panel
    first=first+1
  end if ;first=1

;   zonal 
    res                    = True
    res@gsnDraw            = False             ; don't draw yet
    res@gsnFrame           = False             ; don't advance frame yet
    res@xyMonoLineColor    = False             ; want colored lines
    res@xyLineThicknesses = (/3.,3.,3.,3./)
    res@xyLineColors      = (/"black","blue","red","green"/)  ; change line color
   ;res@xyDashPatterns   = (/0.,0.,0.,0./)      ; make all lines solid
   ;res@tiMainString      = "Zonal Mean "+var      ; add title
   ;res@vpHeightF        = 0.4               ; change aspect ratio of plot
   ;res@vpWidthF         = 0.7
   ;res@trXMinF          = 1890              ; set x-axis minimum
   ;res@tiYAxisString    = "~F35~J~F~C"      ; add an axis title
   ;res@txFontHeightF    = 0.0195            ; change title font heights
   ;delete(res@xyLineThicknesses)      ; Don't want this resource any more
   ;res@xyMarkLineMode = "MarkLines"               ; Markers *and* lines
   ;res@xyMarkers      = (/6,11,16/)               ; 3 different markers
   ;res@xyMarkerColors := (/"blue","red","green"/) ; 3 different colors
 ;  res@vpXF      = vpx
 ;  res@vpYF      = vpy
 ;  res@vpHeightF = vph
 ;  res@vpWidthF  = vpw

; legend
;   res@trXMinF = -60
;   res@trXMaxF =  60
;   res@trYMaxF =0.6
;   res@trYMinF =0.

    res@trYMinF          =-6.5
    res@trYMaxF          = 0.
  ; if (mon.eq.9) then
  ;   res@trYMinF          =-5
  ;   res@trYMaxF          = 0.
  ; end if
;   res@gsnCenterString = "Zonal Mean (W/m~S~2~N~)"
    res@gsnRightString = ""

    res@pmLegendDisplayMode    = "Always"            ; turn on legend
    res@pmLegendSide           = "Top"               ; Change location of 
    res@pmLegendParallelPosF   = .27                  ; move units right
    res@pmLegendOrthogonalPosF = -.8               ; move units down
    res@pmLegendWidthF         = 0.10                ; Change width and
    res@pmLegendHeightF        = 0.10                ; height of legend.
    res@lgPerimOn              = False               ; turn off box around
    res@lgLabelFontHeightF     = .02                 ; label font height
    res@xyExplicitLegendLabels = (/"ProgAer minus NoAer","ClimAer minus NoAer","ProgAer AOD","ClimAer AOD"/)         ; create explicit labels
    res@xyLineColors           = (/"red","blue","orange","cyan"/)
    res@xyLineThicknessF       = 2.5
    res@xyDashPatterns         = (/0,0/)

; resources for "left" variable
  resL                        = res
  resL@tiYAxisString = "RF TOA Diff (W/m~S~2~N~) [solid]"  ; axis string

; resources for "right" variable
  resR                        = res
  resR@xyDashPatterns         = (/1,1/)  ; dashed line for 2nd
  resR@xyLineColors           = (/"red","blue","lightpink","lightblue"/)
  resR@xyLineColors           = (/"lightpink","lightblue","red","blue"/)
  resR@tiYAxisString = "Neg. AOD [dash]"   ; axis string
  resR@xyExplicitLegendLabels = (/"ProgAer AOD","ClimAer AOD","ProgAer AOD","ClimAer AOD"/)
  resR@pmLegendParallelPosF   = .19                  ; move units right
  resR@pmLegendOrthogonalPosF =  -.40               ; move units down
  resR@pmLegendOrthogonalPosF =  -1.0               ; move units down
  resR@gsnCenterString = amon
  resR@trYMinF          =-.46
  resR@trYMaxF          = .0
  ; if (mon.eq.9) then
  ;   resR@trYMinF          =-.4
  ;   resR@trYMaxF          = 0.
  ; end if
  plot(nn) = gsn_csm_xy2(wks,x90&lat,znl_dtoa,znl_aod,resL,resR) ; create plot

  nn=nn+1

  ; res@vpWidthF      = 0.8
  ; res@vpHeightF     = 0.4
  end do ;yr
  end do ;mo

;************************************************
; create panel
;************************************************
  resP                  = True                   ; modify the panel plot
 ;resP@gsnPanelBottom   = 0.05                   ; add space at bottom
 ;resP@gsnPanelFigureStrings= (/"a)","b)","c)"/) ; add strings to panel
 ;resP@amJust   = "TopLeft"
    resP@gsnPanelMainFontHeightF = 0.015
   ;resP@gsnPanelMainString = var+" Monthly Mean 2003-2019 "+amon
    resP@gsnPanelMainString = " Zonal Mean 2003-2019 "
    resP@gsnPanelLabelBar = True                   ; add common colorbar
    resP@lbLabelFontHeightF  = 0.015               ; make labels smaller
    if (do_aod.eq.1) then
      gsn_panel(wks,plot,(/2,2/),resP)               ; now draw as one plot
    else
      gsn_panel(wks,plot,(/2,1/),resP)               ; now draw as one plot
    end if
  end
