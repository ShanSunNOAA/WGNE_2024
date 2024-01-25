;*************************************************
; corel_3.ncl
;
; Concepts illustrated:
;   - Calculating a two-dimensional correlation in time
;   - Reordering an array
;   - Copying attributes from one variable to another
;   - Copying coordinate arrays from one variable to another
;   - Removing the end boxes in a labelbar
;
;************************************************
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
begin
;************************************************
; open file and read in variable
;***********************************************

  mon=5
  narea=0
  narea=1  ;Fig.13
  narea=2  ;Fig.14
  path1 = "/scratch2/BMC/gsd-fv3-dev/sun/post_p8_f90/acc_3var/"

  sfcdata1= "acc_noa0_t2m_"+sprinti("%0.2i",mon)+".nc"
  fin1= addfile(path1+sfcdata1,"r");
  t11 = fin1->t2m(0,:,:);
  t15 = fin1->t2m(4,:,:);

  sfcdata1= "acc_clim_t2m_"+sprinti("%0.2i",mon)+".nc"
  fin1= addfile(path1+sfcdata1,"r");
  t21 = fin1->t2m(0,:,:);
  t25 = fin1->t2m(4,:,:);

  sfcdata1= "acc_chm_t2m_"+sprinti("%0.2i",mon)+".nc"
  fin1= addfile(path1+sfcdata1,"r");
  t31 = fin1->t2m(0,:,:);
  t35 = fin1->t2m(4,:,:);

  sfcdata1= "acc_noa0_precip_"+sprinti("%0.2i",mon)+".nc"
  fin1= addfile(path1+sfcdata1,"r");
  p11 = fin1->precip(0,:,:);
  p15 = fin1->precip(4,:,:);

  sfcdata1= "acc_clim_precip_"+sprinti("%0.2i",mon)+".nc"
  fin1= addfile(path1+sfcdata1,"r");
  p21 = fin1->precip(0,:,:);
  p25 = fin1->precip(4,:,:);

  sfcdata1= "acc_chm_precip_"+sprinti("%0.2i",mon)+".nc"
  fin1= addfile(path1+sfcdata1,"r");
  p31 = fin1->precip(0,:,:);
  p35 = fin1->precip(4,:,:);

  sfcdata1= "acc_noa0_h500_"+sprinti("%0.2i",mon)+".nc"
  fin1= addfile(path1+sfcdata1,"r");
  h11 = fin1->h500(0,:,:);
  h15 = fin1->h500(4,:,:);

  sfcdata1= "acc_clim_h500_"+sprinti("%0.2i",mon)+".nc"
  fin1= addfile(path1+sfcdata1,"r");
  h21 = fin1->h500(0,:,:);
  h25 = fin1->h500(4,:,:);

  sfcdata1= "acc_chm_h500_"+sprinti("%0.2i",mon)+".nc"
  fin1= addfile(path1+sfcdata1,"r");
  h31 = fin1->h500(0,:,:);
  h35 = fin1->h500(4,:,:);

;************************************************
 ;wks = gsn_open_wks("x11", "contour_map")   ;-- open a workstation
  wks = gsn_open_wks("png",get_script_prefix_name())
  plot = new(18,graphic) ; create a plot panel
 
  res                     = True                ; make plot mods
  res@gsnDraw             = False            ; don't draw
  res@gsnFrame            = False            ; don't advance frame
  res@cnInfoLabelOn       = False            ; turn off cn info label
  res@cnFillOn            = True             ; turn on color
 ;cmap = read_colormap_file("precip2_17lev") 
 ;res@cnFillPalette       = cmap(:ncmap-3,:) ; skip last two colors
  res@lbLabelBarOn      = False    ; turn off individual cb's
  res@cnLinesOn         = False    ; turn off contour lines
  res@cnLineLabelsOn    = False    ; turn off line labels

 if (narea.eq.0) then
  res@mpMaxLatF                   = 60       
  res@mpMinLatF                   = -60
  area0=""
 end if

 if (narea.eq.1) then
  res@mpMaxLatF                   = 45           ; N. Africa
  res@mpMinLatF                   = 0
  res@mpMaxLonF                   = 45
  res@mpMinLonF                   = -30
  area0="over N. Africa"
 end if

 if (narea.eq.2) then
  res@mpMaxLatF                   = 45           ; E. Asia
  res@mpMinLatF                   = 0
  res@mpMaxLonF                   = 130
  res@mpMinLonF                   = 70
  area0="over E. Asia"
 end if

 if (narea.eq.3) then
  res@mpMaxLatF                   = 0           ; S. Africa
  res@mpMinLatF                   = -30
  res@mpMaxLonF                   = 45
  res@mpMinLonF                   = -10
  area0="over S. Africa"
 end if

 if (narea.eq.4) then
  res@mpMaxLatF                   = 0           ; choose subregion
  res@mpMinLatF                   = -60
  res@mpMaxLonF                   = 330
  res@mpMinLonF                   = 271
  area0="over S. America"
 end if

  res@cnLevelSelectionMode = "ExplicitLevels"   ; set explicit contour levels
  res@cnFillPalette = "nrl_sirkes";"temp_diff_18lev"; "precip3_16lev"; "sunshine_9lev"; "temp_diff_18lev"; "ncl_default"
 ;res@cnFillPalette        = "BlWhRe"            ; set color map
 ;res@cnFillPalette        = "precip2_17lev"     ; 18 shades, auto contour
 ;res@cnFillPalette        = "MPL_YlOrRd"
 ;res@cnFillPalette        = "precip3_16lev"
  res@cnFillPalette        = "perc2_9lev"

  res@cnFillPalette = "precip2_17lev";
 ;res@cnLevels = (/.1,.2,.3,.4,.5,.6,.7,.8,.9/)  ; 9
 ;res@cnFillColors= (/2,4,6,8,10,12,13,14,15,16/)  ; 10 precip2_17lev
  res@cnLevels = (/.05,.15,.25,.35,.45,.55,.65,.75,.85,.95/)  ; 10
  res@cnFillColors= (/0,2,4,6,8,10,12,13,14,15,16/)  ; 11 precip2_17lev

  res@gsnLeftString = ""
  res@gsnRightString = ""

 ;res@cnLevelSelectionMode = "ManualLevels"      ; manually set cn levels
 ;res@cnMinLevelValF       = 5.0                ; set min contour level
 ;res@cnMaxLevelValF       = 90.0               ; set max contour level
 ;res@cnLevelSpacingF      = 5.                ; set contour spacing

; res@cnLabelBarEndStyle   = "ExcludeOuterBoxes" ; remove the two end boxes from the labelbar
; res@pmLabelBarWidthF     = 0.8
; res@cnLinesOn            = False               ; turn off contour lines
; res@cnLevelSelectionMode = "ManualLevels"      ; manually set cn levels

 ;res@tiMainString         = "RMSE Monthly ORL 2003-2019 May Ctrl"
  res@gsnCenterStringFontHeightF = 0.04

  res@gsnCenterString = "T2m NoAer wk1"
  plot(0) = gsn_csm_contour_map(wks,t11,res)
  res@gsnCenterString = "T2m NoAer wk34"
  plot(1) = gsn_csm_contour_map(wks,t15,res)

  res@gsnCenterString = "Prcp NoAer wk1"
  plot(2) = gsn_csm_contour_map(wks,p11,res)
  res@gsnCenterString = "Prcp NoAer wk34"
  plot(3) = gsn_csm_contour_map(wks,p15,res)

  res@gsnCenterString = "H500 NoAer wk1"
  plot(4) = gsn_csm_contour_map(wks,h11,res)
  res@gsnCenterString = "H500 NoAer wk34"
  plot(5) = gsn_csm_contour_map(wks,h15,res)

  res@gsnCenterString = "T2m ClimAer wk1"
  plot(6) = gsn_csm_contour_map(wks,t21,res)
  res@gsnCenterString = "T2m ClimAer wk34"
  plot(7) = gsn_csm_contour_map(wks,t25,res)

  res@gsnCenterString = "Prcp ClimAer wk1"
  plot(8) = gsn_csm_contour_map(wks,p21,res)
  res@gsnCenterString = "Prcp ClimAer wk34"
  plot(9) = gsn_csm_contour_map(wks,p25,res)

  res@gsnCenterString = "H500 ClimAer wk1"
  plot(10) = gsn_csm_contour_map(wks,h21,res)
  res@gsnCenterString = "H500 ClimAer wk34"
  plot(11) = gsn_csm_contour_map(wks,h25,res)

  res@gsnCenterString = "T2m ProgAer wk1"
  plot(12) = gsn_csm_contour_map(wks,t31,res)
  res@gsnCenterString = "T2m ProgAer wk34"
  plot(13) = gsn_csm_contour_map(wks,t35,res)

  res@gsnCenterString = "Prcp ProgAer wk1"
  plot(14) = gsn_csm_contour_map(wks,p31,res)
  res@gsnCenterString = "Prcp ProgAer wk34"
  plot(15) = gsn_csm_contour_map(wks,p35,res)

  res@gsnCenterString = "H500 ProgAer wk1"
  plot(16) = gsn_csm_contour_map(wks,h31,res)
  res@gsnCenterString = "H500 ProgAer wk34"
  plot(17) = gsn_csm_contour_map(wks,h35,res)

  resP                  = True                   ; modify the panel plot
  resP@gsnPanelLabelBar = True                  ; add common colorbar
  resP@lbLabelFontHeightF  = 0.012               ; make labels smaller
 ;resP@gsnPanelBottom   = 0.05                   ; add space at bottom
 ;resP@gsnPanelFigureStrings= (/"a)","b)","c)"/) ; add strings to panel
  resP@gsnPanelMainFontHeightF = 0.015
  if (mon.eq.5) then 
    resP@gsnPanelMainString = "ACC at wk1 & wk34 May 2003-2019 "+area0
  end if
  if (mon.eq.9) then 
    resP@gsnPanelMainString = "ACC at wk1 & wk34 Sept 2003-2019 "+area0
  end if

  gsn_panel(wks,plot,(/3,6/),resP)               ; now draw as one plot
  
;************************************************
end
