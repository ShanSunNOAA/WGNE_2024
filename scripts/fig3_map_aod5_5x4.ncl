;---------------------------------------------------------------
;  NCL User Guide Example: NUG_triangular_grid_ICON.ncl
;
;  Grid type:         ICON - Unstructured grid
;
;  Settings:          sub-region,
;                     manual-levels,
;                     draw colored triangles with outlines,
;                     don't draw missing values
;  KMF 31.10.14
;---------------------------------------------------------------

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin

  yr1=2019
  yr2=2019
  yr1=9999
  yr2=9999
  mo1=5
  mo2=9
  dmo=mo2-mo1
  if (dmo.eq.0) then
    dmo=1
  end if
  gl=1
  bias=0

; only 1 can be true
  do_aod=1
  cld=0 
  toa=0
  net=0
  sw=0
  modis_aod_ave=new(17,float)
  znlx=new((/6,7/),float)

  path0 = "/scratch1/BMC/gsd-fv3-dev/data_others/AOD/monthlymean/"
  path0 = "/scratch1/BMC/gsd-fv3-dev/data_others/AOD/merra2/"
  pth_chm = "/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/COMROOT/chm0/"
  pth_clim ="/scratch2/BMC/gsd-hpcs/Shan.Sun/clim_togo_exp/COMROOT/clim0/"
  nc_mr=0 ; monthly
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
  mmm=0
  do mon=mo1,mo2,dmo
    if (mon.eq.5) then
      amon="May"
    end if
    if (mon.eq.9) then
      amon="Sep"
    end if
    patha=path0
    pathb=pth_chm
  do yr=yr1,yr2
    if (yr.eq.9999) then
      modis="MODIS_"+amon+"_2003_2019_ave.nc"
      merra="MERRA2_400.tavgM_2d_aer_Nx."+amon+"_2003_2019_ave.nc4"
      ceres="CERES_2003_2019_"+amon+"_ave.nc"
      myr=" 17yr "
    else
      modis="MODIS_"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".nc"
      merra="MERRA2_400.tavgM_2d_aer_Nx."+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".nc4"
      ceres="CERES_"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".nc"
      myr=sprinti("%0.4i",yr)
    end if
    sfcdata1= "gfs."+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+"01/00/var_wk_1x1.nc"

    var=" AOD "
    obslb=" MODIS "
    obslb=" MERRA2 "
    fin0= addfile(path0+merra,"r");
   ;print(fin0)
    printVarSummary(fin0)
    fpng= "aod_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
   ;if (gl.eq.1) then
   ;  obs=fin0->MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean(lat1:lat2,lon1:lon2); monthly
   ;  modis_file="/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/glb_aod/aod_gl_modis_2003_2019_"+sprinti("%0.2i",mon)+"_sht.txt"
   ;else
   ;  obs=fin0->MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean(lat1:lat2,lon1+180:lon2+180); monthly
   ;  modis_file="/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/glb_aod/aod_gl_modis_2003_2019_"+sprinti("%0.2i",mon)+"_sht.txt"
   ;end if
   ;modis_aod_ave=asciiread(modis_file,(/17/),"float")

    fin2= addfile(pathb+sfcdata1,"r");
    if (gl.eq.1) then
      xall =fin0->DUEXTTAU(nc_mr,:,:); monthly
      xdust=fin0->DUEXTTAU(nc_mr,:,:); monthly
      xseas=fin0->SSEXTTAU(nc_mr,:,:); monthly
      xsulf=fin0->SUEXTTAU(nc_mr,:,:); monthly
      xoc  =fin0->OCEXTTAU(nc_mr,:,:); monthly
      xbc  =fin0->BCEXTTAU(nc_mr,:,:); monthly

      yall =fin2->maod(ncrec4,:,:); monthly
      ydust=fin2->maoddust(ncrec4,:,:); monthly
      yseas=fin2->maodseas(ncrec4,:,:); monthly
      ysulf=fin2->maodsulf(ncrec4,:,:); monthly
      yoc  =fin2->maodoc(ncrec4,:,:); monthly
      ybc  =fin2->maodbc(ncrec4,:,:); monthly

      lata=fin0->lat;
      latb=fin2->lat;
    else
      xall =fin0->DUEXTTAU(nc_mr,lat1:lat2,lon1:lon2); monthly
      xdust=fin0->DUEXTTAU(nc_mr,lat1:lat2,lon1:lon2); monthly
      xseas=fin0->SSEXTTAU(nc_mr,lat1:lat2,lon1:lon2); monthly
      xsulf=fin0->SUEXTTAU(nc_mr,lat1:lat2,lon1:lon2); monthly
      xoc  =fin0->OCEXTTAU(nc_mr,lat1:lat2,lon1:lon2); monthly
      xbc  =fin0->BCEXTTAU(nc_mr,lat1:lat2,lon1:lon2); monthly

      yall =fin2->maod(ncrec4,lat1:lat2,lon1:lon2); monthly
      ydust=fin2->maoddust(ncrec4,lat1:lat2,lon1:lon2); monthly
      yseas=fin2->maodseas(ncrec4,lat1:lat2,lon1:lon2); monthly
      ysulf=fin2->maodsulf(ncrec4,lat1:lat2,lon1:lon2); monthly
      yoc  =fin2->maodoc(ncrec4,lat1:lat2,lon1:lon2); monthly
      ybc  =fin2->maodbc(ncrec4,lat1:lat2,lon1:lon2); monthly

      lata=fin0->lat(lat1:lat2);
      latb=fin2->lat(lat1:lat2);
    end if

    clat0  = cos(lata*rad)
    clat0!0="lat"  ; name the dim
    clat0&lat=lata ; assign the coord var

    clat2  = cos(latb*rad)
    clat2!0="lat"  ; name the dim
    clat2&lat=latb ; assign the coord var

   ;obs@_FillValue = -9999

   ;ave_x0 = wgt_areaave(obs, clat, 1.0, 1)
    ave_xa = wgt_areaave(xall,clat0, 1.0, 1)
    ave_xd = wgt_areaave(xdust,clat0, 1.0, 1)
    ave_xs = wgt_areaave(xseas,clat0, 1.0, 1)
    ave_xf = wgt_areaave(xsulf,clat0, 1.0, 1)
    ave_xo = wgt_areaave(xoc,clat0, 1.0, 1)
    ave_xb = wgt_areaave(xbc,clat0, 1.0, 1)

   ;ave_y0 = wgt_areaave(obs, clat2, 1.0, 1)
    ave_ya = wgt_areaave(yall,clat2, 1.0, 1)
    ave_yd = wgt_areaave(ydust,clat2, 1.0, 1)
    ave_ys = wgt_areaave(yseas,clat2, 1.0, 1)
    ave_yf = wgt_areaave(ysulf,clat2, 1.0, 1)
    ave_yo = wgt_areaave(yoc,clat2, 1.0, 1)
    ave_yb = wgt_areaave(ybc,clat2, 1.0, 1)

    print(ave_xd)
   ;print(modis_aod_ave)
    if (yr.lt.9999) then
      ave_x0=modis_aod_ave(yr-2003)
    else
      ave_x0=avg(modis_aod_ave)
    end if

    n0=-70 ; =>70S
   ;printVarSummary(xall)
   ;printVarSummary(clat)
   ;print(znlx)
    fpng= "bar_znl_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"

  if (first.eq.1) then
    wks_type = "png"
    wks_type@wkWidth = 2500
    wks_type@wkHeight = 2500
    wks = gsn_open_wks(wks_type,get_script_prefix_name())
   ;wks = gsn_open_wks("pdf",get_script_prefix_name())
    plot = new(20,graphic) ; create a plot panel
    res                     = True
    res@gsnDraw             = False            ; don't draw
    res@gsnFrame            = False            ; don't advance frame
    res@cnInfoLabelOn       = False            ; turn off cn info label
    res@cnFillOn            = True             ; turn on color
    res@mpFillOn            = False             ; turn on color
    res@mpOutlineOn = True 
    res@lbLabelBarOn        = False
    res@lbLabelFontHeightF  = 0.03   ; make labels smaller
    res@cnLinesOn         = False    ; turn off contour lines
    res@cnLineLabelsOn    = False    ; turn off line labels
    res@cnLevelSelectionMode = "ExplicitLevels"
    res@gsnLeftString     = ""
    res@cnFillPalette     = "precip3_16lev"     ; set color map
   ;res@cnLevels    = (/.1,.2,.3,.4,.5,.6,.7,.8,.9/)   ; AOD
   ;res@cnLevels    = (/.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.60,.65,.7,.75/)   ; 15 AOD
   ;res@cnLevels    = (/.03,.06,.09,.12,.15,.18,.21,.24,.27,.30,.33,.36,.39,.42,.45/)   ; 15 AOD
   ;res@cnLevels    = (/.04,.08,.12,.16,.20,.24,.28,.32,.36,.40,.44,.48,.52,.56,.60/)   ; 15 AOD
   ;res@cnFillColors = (/-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16/) ; 16 colors
    res@cnLevels    = (/.05,.10,.15,.20,.25,.30,.35,.40,.45,.50/) 
    res@cnFillColors = (/-1,2,4,6,8,10,12,13,14,15,16/)
    res@gsnCenterStringFontHeightF = 0.040
    res@gsnRightStringFontHeightF  = 0.038
  ; res@pmTickMarkDisplayMode = "Always"            ; turn on built-in tickmarks
   ;res@tiMainString= ""
    res@tmXTOn = False   ; Turn off tick marks on the x-axis
    res@tmYLOn = False   ; Turn off tick marks on the y-axis

    first=first+1
  end if ;first=1

    res@tiMainString= ""
    res@mpProjection = "Robinson"

    res@gsnRightString = sprintf("%4.2f",ave_xf)
    res@gsnCenterString = " Sulfate"
    plot(mmm) = gsn_csm_contour_map(wks,xsulf,res)

    res@gsnRightString = sprintf("%4.2f",ave_xd)
    res@gsnCenterString = "Dust"
    plot(mmm+4) = gsn_csm_contour_map(wks,xdust,res)

    ave_xb=ave_xb*5.
    xbc=xbc*5.
    res@gsnRightString = sprintf("%4.2f",ave_xb)
    res@gsnCenterString = "BCx5"
    plot(mmm+8) = gsn_csm_contour_map(wks,xbc,res)

    res@gsnRightString = sprintf("%4.2f",ave_xo)
    res@gsnCenterString = "OC"
    plot(mmm+12) = gsn_csm_contour_map(wks,xoc,res)

    res@gsnRightString = sprintf("%4.2f",ave_xs)
    res@gsnCenterString = "Sea Salt"
    plot(mmm+16) = gsn_csm_contour_map(wks,xseas,res)

    res@gsnRightString = sprintf("%4.2f",ave_yf)
    res@gsnCenterString = " Sulfate"
    plot(mmm+1) = gsn_csm_contour_map(wks,ysulf,res)

    res@gsnRightString = sprintf("%4.2f",ave_yd)
    res@gsnCenterString = "Dust"
    plot(mmm+5) = gsn_csm_contour_map(wks,ydust,res)

    ave_yb=ave_yb*5.
    ybc=ybc*5.
    res@gsnRightString = sprintf("%4.2f",ave_yb)
    res@gsnCenterString = "BCx5"
    plot(mmm+9) = gsn_csm_contour_map(wks,ybc,res)

    res@gsnRightString = sprintf("%4.2f",ave_yo)
    res@gsnCenterString = "OC"
    plot(mmm+13) = gsn_csm_contour_map(wks,yoc,res)

    res@gsnRightString = sprintf("%4.2f",ave_ys)
    res@gsnCenterString = "Sea Salt"
    plot(mmm+17) = gsn_csm_contour_map(wks,yseas,res)
    mmm=mmm+2
  end do ;yr
  end do ;mo

;************************************************
; Attach text to plot using plot coordinates.
;************************************************
; txres               = True                     ; text mods desired
; txres@txFontHeightF = 0.03                     ; font smaller. default big
; dum = gsn_add_text(wks,plot(0),"Text0",10,0.0,txres) 
; dum = gsn_add_text(wks,plot(1),"Text1",10,0.0,txres) 
; dum = gsn_add_text(wks,plot(2),"Text2",10,0.0,txres) 
; dum = gsn_add_text(wks,plot(3),"Text3",10,0.0,txres) 

;************************************************
; create panel
;************************************************
  resP                  = True                   ; modify the panel plot
  resP@gsnPanelMainFontHeightF = 0.009
  resP@gsnPanelLabelBar = True                   ; add common colorbar
  resP@pmLabelBarWidthF = 0.45
  resP@gsnPanelMainPosYF= 0.52 
 ;resP@gsnPanelTop = 1.0 ; Default=1
 ;resP@gsnPanelFigureStrings= (/"a)","b)","c)"/) ; add strings to panel
 ;resP@amJust   = "TopLeft"
  resP@gsnPanelMainString= "MERRA2 May                       ProgAer May                            MERRA2 Sep                        ProgAer Sep"
  resP@gsnPanelYWhiteSpacePercent = .1
  resP@lbLabelFontHeightF  = 0.009         
  gsn_panel(wks,plot,(/5,4/),resP)               ; now draw as one plot

  end
