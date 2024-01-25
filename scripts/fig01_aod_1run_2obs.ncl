load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin

  yr1=2003
  yr2=2003
  yr1=9999
  yr2=9999
  mo1=5
  mo2=9
  dmo=mo2-mo1
  gl=1
  bias=0

; only 1 can be true
  do_aod=1
  cld=0 
  toa=0
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
  rad = 4.0*atan(1.0)/180.0
  nn=0
  do mon=mo1,mo2,dmo
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
      merra="MERRA2_400.tavgM_2d_aer_Nx."+amon+"_2003_2019_ave.nc4"
    else
      modis="MODIS_"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".nc"
      ceres="CERES_"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".nc"
    end if
    sfcdata1= "gfs."+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+"01/00/var_wk_1x1.nc"
    fin1= addfile(patha+sfcdata1,"r");
   ;fin2= addfile(pathb+sfcdata1,"r");
   ;fin3= addfile(pathc+sfcdata1,"r");
    lona=fin1->lon(lon1:lon2);
    lata=fin1->lat(lat1:lat2);
    clat  = cos(lata*rad)

    fout0= "htfx_gl_ceres_"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".txt"
    fout1= "htfx_gl_chem"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".txt"
    fout2= "htfx_gl_gf_no_aer"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".txt"
    fout3= "htfx_gl_gf_aer"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".txt"

   if (do_aod.eq.1 ) then
    var=" AOD "
    obs1=" MODIS "
    obs2=" MERRA2 "
    pth_md = "/scratch1/BMC/gsd-fv3-dev/data_others/AOD/monthlymean/"
    pth_mr = "/scratch1/BMC/gsd-fv3-dev/data_others/AOD/merra2/"

    fin01= addfile(pth_md+modis,"r");
    fin02= addfile(pth_mr+merra,"r");
    fpng= "aod_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    if (gl.eq.1) then
      x90=fin01->MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean(lat1:lat2,lon1:lon2); monthly
      modis_file="/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/glb_aod/aod_gl_modis_2003_2019_"+sprinti("%0.2i",mon)+"_sht.txt"
      x92=fin02->TOTEXTTAU(0,:,:); monthly
      latb=fin02->lat(:);
      clat2 = cos(latb*rad)
    else
      x90=fin0->MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean(lat1:lat2,lon1+180:lon2+180); monthly
      modis_file="/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/glb_aod/aod_gl_modis_2003_2019_"+sprinti("%0.2i",mon)+"_sht.txt"
    end if
    modis_aod_ave=asciiread(modis_file,(/17/),"float")
    x91=fin1->maod(ncrec4,lat1:lat2,lon1:lon2); monthly
    x93=fin1->maod(ncrec4,lat1:lat2,lon1:lon2); monthly

    x90@_FillValue = -9999

    ave_x0 = wgt_areaave(x90, clat, 1.0, 1)
    ave_x1 = wgt_areaave(x91, clat, 1.0, 1)
    ave_x2 = wgt_areaave(x92, clat2, 1.0, 1)
    ave_x3 = wgt_areaave(x93, clat, 1.0, 1)
    print(modis_aod_ave)
    if (yr.lt.9999) then
      ave_x0=modis_aod_ave(yr-2003)
    else
      ave_x0=avg(modis_aod_ave)
    end if
 end if ; do_aod=1
   print("ave_x0="+ave_x0+" ave_x1="+ave_x1+" ave_x2="+ave_x2+" ave_x3="+ave_x3)

  if (first.eq.1) then
   ;wks = gsn_open_wks("x11", "contour_map")   ;-- open a workstation
    wks = gsn_open_wks("png",get_script_prefix_name())
    if (do_aod.eq.1) then
      plot = new(6,graphic) ; create a plot panel
    else
      plot = new(8,graphic) ; create a plot panel
    end if
    first=first+1
  end if ;first=1

    res                     = True
    res@gsnDraw             = False            ; don't draw
    res@gsnFrame            = False            ; don't advance frame
    res@cnInfoLabelOn       = False            ; turn off cn info label
    res@cnFillOn            = True             ; turn on color
    res@mpFillOn            = False             ; turn on color
    res@mpOutlineOn = True 
    res@cnFillPalette     = "nrl_sirkes"
    res@lbLabelBarOn      = True
    if (do_aod.eq.1) then
      res@lbLabelBarOn    = False
    end if
    if (bias.eq.0) then
      res@lbLabelBarOn    = False
    end if
    res@lbLabelFontHeightF  = 0.03   ; make labels smaller
    res@cnLinesOn         = False    ; turn off contour lines
    res@cnLineLabelsOn    = False    ; turn off line labels
   ;res@tiMainString      = ""
    res@gsnLeftString = ""
    res@gsnRightString = ""

;   without these, the map starts at 180E. Otherwise it would start at 0E.
;   res@mpMinLatF    = lat1-90;
;   res@mpMaxLatF    = lat2-90;
;   res@mpMinLonF    = lon1
;   res@mpMaxLonF    = lon2

    res@gsnCenterStringFontHeightF = 0.030
    res@gsnRightStringFontHeightF = 0.030
  ; res@cnLevelSelectionMode = "ManualLevels"
    res@cnLevelSelectionMode = "ExplicitLevels"

    if (yr.lt.9999) then
      res@gsnCenterString = amon+sprinti("%0.4i",yr)+var+obs1
    else
      res@gsnCenterString = amon+obs1  ; +var+obs
    end if

    res@cnFillPalette  = "precip3_16lev"
    res@cnLevels    = (/.1,.2,.3,.4,.5,.6,.7,.8,.9/)   ; AOD
    res@cnFillColors= (/-1,1,2,4,6,8,10,12,14,16/)

    res@gsnRightString = sprintf("%4.2f",ave_x0)
    plot(nn) = gsn_csm_contour_map(wks,x90,res)

    res@gsnCenterString = amon+" MERRA2"
    res@gsnRightString = sprintf("%4.2f",ave_x2)
    plot(nn+1) = gsn_csm_contour_map(wks,x92,res)

    res@gsnCenterString = amon+" ProgAer"
    res@gsnRightString = sprintf("%4.2f",ave_x1)
    plot(nn+2) = gsn_csm_contour_map(wks,x91,res)
    nn=nn+3

  end do ;yr
  end do ;mo

;************************************************
; create panel
;************************************************
    resP                  = True                   ; modify the panel plot
    resP@gsnPanelLabelBar = False                   ; add common colorbar
    resP@gsnPanelMainFontHeightF = 0.020
    resP@gsnPanelMainFontHeightF = 0.012
    resP@gsnPanelMainString = var+"Monthly Mean 2003-2019 "
    if (do_aod.eq.1) then
      resP@gsnPanelLabelBar = True                   ; add common colorbar
    end if
    if (bias.eq.0) then
      resP@gsnPanelLabelBar = True                   ; add common colorbar
    end if
    resP@lbLabelFontHeightF  = 0.015               ; make labels smaller
    if (do_aod.eq.1) then
      gsn_panel(wks,plot,(/2,3/),resP)               ; now draw as one plot
    else
      gsn_panel(wks,plot,(/2,4/),resP)               ; now draw as one plot
    end if
  end
