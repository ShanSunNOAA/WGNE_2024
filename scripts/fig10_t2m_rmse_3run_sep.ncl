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
  rad    = 4.0*atan(1.0)/180.0
  bar_share=0
  runid1="ProgAer "
  runid2="ClimAer "
  runid3="NoAer "

  do nn=1,1

  if (nn.eq.0) then
    mm=5
    amon="IC=May1"
  end if
  if (nn.eq.1) then
    mm=9
    amon=""
  end if
  if (mm.eq.5) then
    file0=addfile("/scratch1/BMC/gsd-fv3-dev/data_others/era5/era5_05_4d.nc","r")
    file1=addfile("/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm0/var_05_4d.nc","r")
    file2=addfile("/scratch1/BMC/wrfruc/Shan.Sun/wgne_clim/clim0/var_05_4d.nc","r")
  end if
  if (mm.eq.9) then
    file0=addfile("/scratch1/BMC/gsd-fv3-dev/data_others/era5/era5_09_4d.nc","r")
    file1=addfile("/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm0/var_09_4d.nc","r")
    file2=addfile("/scratch1/BMC/wrfruc/Shan.Sun/wgne_clim/clim0/var_09_4d.nc","r")
    file3=addfile("/scratch1/BMC/wrfruc/Shan.Sun/wgne_noa0/noa00/var_09_4d.nc","r")
  end if
  latn=file1->lat
  clat=cos(latn*rad)

  read0=file0->t2m(:,2:3,:,:)
  read1=file1->t2m(:,2:3,:,:)
  read2=file2->t2m(:,2:3,:,:)
  read3=file3->t2m(:,2:3,:,:)

  x0=file0->t2m(:,0,:,:); filler
  x1=file0->t2m(:,0,:,:); filler
  x2=file0->t2m(:,0,:,:); filler
  x3=file0->t2m(:,0,:,:); filler

  rmse1=file0->t2m(0,0,:,:); filler
  rmse2=file0->t2m(0,0,:,:); filler
  rmse3=file0->t2m(0,0,:,:); filler
  prob12=file0->t2m(0,0,:,:); filler
  prob13=file0->t2m(0,0,:,:); filler
 
  read0@_FillValue = -9999
  read1@_FillValue = -9999
  read2@_FillValue = -9999
  read3@_FillValue = -9999

  printVarSummary(read0)
  printMinMax(read0, True)
  printVarSummary(read1)
  printMinMax(read1, True)

  x0=dim_avg_n_Wrap(read0,1) ; monthly or wk34 mean
  x1=dim_avg_n_Wrap(read1,1) ; monthly or wk34 mean
  x2=dim_avg_n_Wrap(read2,1) ; monthly or wk34 mean
  x3=dim_avg_n_Wrap(read3,1) ; monthly or wk34 mean
  printVarSummary(x0)
  printMinMax(x0, True)
  printVarSummary(x1)
  printMinMax(x1, True)

  rmse1=dim_rmsd_n_Wrap(x1,x0,0)
  rmse2=dim_rmsd_n_Wrap(x2,x0,0)
  rmse3=dim_rmsd_n_Wrap(x3,x0,0)
  ave_x1=wgt_areaave_Wrap(rmse1,clat, 1.0, 0)
  ave_x2=wgt_areaave_Wrap(rmse2,clat, 1.0, 0)
  ave_x3=wgt_areaave_Wrap(rmse3,clat, 1.0, 0)
  print(ave_x1)
  print(ave_x2)
  print(ave_x3)

;;;;;;;;;   prob12    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  xtmp = x1(lat|:,lon|:,time|:)       ; reorder but do it only once [temporary]
  ytmp = x2(lat|:,lon|:,time|:)
    
  xAve = dim_avg (xtmp)              ; calculate means at each grid point 
  yAve = dim_avg (ytmp)
  xVar = dim_variance (xtmp)         ; calculate variances
  yVar = dim_variance (ytmp)

  sigr = 0.05                        ; critical sig lvl for r
  xEqv = equiv_sample_size (xtmp, sigr,0)
  yEqv = equiv_sample_size (ytmp, sigr,0)

  xN   = wgt_areaave (xEqv, clat, 1., 0)    ; wgty could be gaussian weights 
  yN   = wgt_areaave (yEqv, clat, 1., 0) 

  iflag= False                        ; population variance similar
  prob12 = ttest(xAve,xVar,xN, yAve,yVar,yN, iflag, False) 
 ;printVarSummary(rmse1)
  printVarSummary(prob12)
 ;printMinMax(rmse1, True)
  printMinMax(prob12, True)

;;;;;;;;;   prob13    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  xtmp = x1(lat|:,lon|:,time|:)       ; reorder but do it only once [temporary]
  ytmp = x3(lat|:,lon|:,time|:)
    
  xAve = dim_avg (xtmp)              ; calculate means at each grid point 
  yAve = dim_avg (ytmp)
  xVar = dim_variance (xtmp)         ; calculate variances
  yVar = dim_variance (ytmp)

  sigr = 0.05                        ; critical sig lvl for r
  xEqv = equiv_sample_size (xtmp, sigr,0)
  yEqv = equiv_sample_size (ytmp, sigr,0)

  xN   = wgt_areaave (xEqv, clat, 1., 0)    ; wgty could be gaussian weights 
  yN   = wgt_areaave (yEqv, clat, 1., 0) 

  iflag= False                        ; population variance similar
  prob13 = ttest(xAve,xVar,xN, yAve,yVar,yN, iflag, False) 
 ;printVarSummary(rmse1)
  printVarSummary(prob13)
 ;printMinMax(rmse1, True)
  printMinMax(prob13, True)

  wks_type = "png"
  wks_type@wkWidth = 2500
  wks_type@wkHeight = 2500
  if (nn.eq.1) then
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
   ;wks = gsn_open_wks("x11", "contour_map")   ;-- open a workstation
   ;wks = gsn_open_wks("pdf",get_script_prefix_name())
    wks = gsn_open_wks(wks_type,get_script_prefix_name())
   ;wks = gsn_open_wks("eps",get_script_prefix_name())
    plot = new(4,graphic) ; create a plot panel
    res              = True
    res@gsnDraw  = False  ;;; required
    res@gsnFrame = False  ;;; required
    res@gsnMaximize  = True     ; maximize plot in frame
    res@cnFillOn     = True     ; turn on contour fill
    res@cnLinesOn    = False    ; turn off contour fill
    res@cnLevelSelectionMode = "ManualLevels"
    res@gsnLeftString   = " "
    res@gsnRightString  = " "
    res@mpProjection = "Robinson"
    if (bar_share.eq.1) then
      res@lbLabelBarOn    = False
    else
      res@lbLabelBarOn    = True
    end if
    res@gsnRightStringFontHeightF = 0.018
    res@gsnCenterStringFontHeightF = 0.022
    res@lbLabelFontHeightF  = 0.020
    res@tiMainString    = " "
    res@gsnLeftString   = " "
  end if ; nn

  res@gsnCenterString = "T2m "+runid1+amon
  res@gsnRightString = sprintf("%4.2f",ave_x1)
  res@cnFillPalette= "precip2_17lev"  ; white-in-middle
  res@cnMinLevelValF= 0.
  res@cnMaxLevelValF= 12.
  res@cnLevelSpacingF= 1.
  plot(0) = gsn_csm_contour_map(wks,rmse1,res)

  res@gsnCenterString = runid2+amon
  res@gsnRightString = sprintf("%4.2f",ave_x2)
  plot(1) = gsn_csm_contour_map(wks,rmse2,res)

  res@gsnCenterString = runid3+amon
  res@gsnRightString = sprintf("%4.2f",ave_x3)
  plot(2) = gsn_csm_contour_map(wks,rmse3,res)

;;;;;;;;;;;;;;  diff & hatch
  res@gsnCenterString = runid1+"- "+runid2+" (p<0.05 hatched)"
  rmse1=rmse1-rmse2
  res@gsnRightString = sprintf("%4.2f",ave_x1-ave_x2)
  res@cnFillPalette= "CBR_coldhot"  ; white-in-middle
  res@cnMinLevelValF= -2.
  res@cnMaxLevelValF=  2.
  res@cnLevelSpacingF= .2
  plot(3) = gsn_csm_contour_map(wks,rmse1,res) ;diff

  res2 = True
  res2@gsnDraw             = False       ; Do not draw plot
  res2@gsnFrame            = False       ; Do not advance frome
  res2@cnLevelSelectionMode = "ManualLevels" ; set manual contour levels
  res2@cnMinLevelValF      = 0.00        ; set min contour level
  res2@cnMaxLevelValF      = 0.05        ; set max contour level
  res2@cnLevelSpacingF     = 0.05        ; set contour spacing
  res2@cnInfoLabelOn       = False
  res2@cnLinesOn           = False       ; do not draw contour lines
  res2@cnLineLabelsOn      = False       ; do not draw contour labels
  res2@cnFillScaleF        = 0.5         ; add extra density
  res2@gsnLeftString = ""
  res2@gsnRightString = ""
  res2@gsnAddCyclic = True
  res2@cnFillOn = False
  res2@cnFillColor = "white"
  res2@cnFillColor = "blue4"
  res2@cnFillColor = "black"
  res2@cnFillColor = "springgreen"
  res2@cnFillColor = "mediumorchid1"

  iplot = gsn_csm_contour(wks,prob12,res2)
  opt   = True
  opt@gsnShadeFillType = "Pattern"
  opt@gsnShadeLow     = 16 ;17
  iplot = gsn_contour_shade(iplot,0.05, -999, opt) ; shade prob <0.05
  overlay(plot(3),iplot)
  
;;;;;;;;;;;;;;  diff & hatch
; res@gsnCenterString = runid1+"- "+runid3+" (p<0.05 hatched)"
; rmse1=rmse1-rmse3
; res@gsnRightString = sprintf("%4.2f",ave_x1-ave_x3)
; res@cnFillPalette= "CBR_coldhot"  ; white-in-middle
; res@cnMinLevelValF= -4.
; res@cnMaxLevelValF=  4.
; res@cnLevelSpacingF= .4
; plot(4) = gsn_csm_contour_map(wks,rmse1,res) ;diff

; iplot = gsn_csm_contour(wks,prob13,res2)
; opt   = True
; opt@gsnShadeFillType = "Pattern"
; opt@gsnShadeLow     = 16 ;17
; iplot = gsn_contour_shade(iplot,0.05, -999, opt) ; shade prob <0.05
; overlay(plot(4),iplot)
  
  end do ; nn

  resP                  = True                   ; modify the panel plot
  if (bar_share.eq.1) then
    resP@gsnPanelLabelBar = True                  ; add common colorbar
  else
    resP@gsnPanelLabelBar = False                  ; add common colorbar
  end if
  resP@lbLabelFontHeightF  = 0.012               ; make labels smaller
 ;resP@gsnPanelBottom   = 0.05                   ; add space at bottom
 ;resP@gsnPanelFigureStrings= (/"a)","b)","c)"/) ; add strings to panel
  resP@gsnPanelMainFontHeightF = 0.013
; (W/m~S~2~N~)
  resP@gsnPanelMainString = "RMSE wk3+4 against ERA5 IC=1Sep 2003-2019"
  gsn_panel(wks,plot,(/1,4/),resP)               ; now draw as one plot
end
