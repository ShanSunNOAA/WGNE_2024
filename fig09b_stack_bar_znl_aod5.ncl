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
  if (dmo.eq.0) then
    dmo=1
  end if 
  gl=1
  bias=1

; only 1 can be true
  do_aod=0
  cld=1 
  toa=0
  net=0
  sw=0
  modis_aod_ave=new(17,float)

  pth_chm = "/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/COMROOT/chm0/"
  pth_clim ="/scratch2/BMC/gsd-hpcs/Shan.Sun/clim_togo_exp/COMROOT/clim0/"
  ncrec1=0 ; monthly
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
    znl5=new((/6,7/),float)
    znl9=new((/6,7/),float)

  do mon=mo1,mo2,dmo
    if (mon.eq.5) then
      amon="May"
    end if
    if (mon.eq.9) then
      amon="Sep"
    end if
    patha=pth_chm
    pathb=pth_clim
  do yr=yr1,yr2
    if (yr.eq.9999) then
      modis="MODIS_"+amon+"_2003_2019_ave.nc"
      ceres="CERES_2003_2019_"+amon+"_ave.nc"
    else
      modis="MODIS_"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".nc"
      ceres="CERES_"+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+".nc"
    end if
    sfcdata1= "gfs."+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+"01/00/var_wk_1x1.nc"
    fin1= addfile(patha+sfcdata1,"r");
    fin2= addfile(pathb+sfcdata1,"r");
    lonn=fin1->lon(lon1:lon2);
    latn=fin1->lat(lat1:lat2);
    clat   = cos(latn*rad)
    clat!0="lat"  ; name the dim
    clat&lat=latn ; assign the coord var

    var=" AOD "
    obslb=" MODIS "
    path0 = "/scratch1/BMC/gsd-fv3-dev/data_others/AOD/monthlymean/"
    fin0= addfile(path0+modis,"r");
   ;print(fin0)
    fpng= "aod_gl_"+sprinti("%0.4i",yr1)+"_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"
    if (gl.eq.1) then
      obs=fin0->MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean(lat1:lat2,lon1:lon2); monthly
      modis_file="/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/glb_aod/aod_gl_modis_2003_2019_"+sprinti("%0.2i",mon)+"_sht.txt"
    else
      obs=fin0->MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean(lat1:lat2,lon1+180:lon2+180); monthly
      modis_file="/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/glb_aod/aod_gl_modis_2003_2019_"+sprinti("%0.2i",mon)+"_sht.txt"
    end if
    modis_aod_ave=asciiread(modis_file,(/17/),"float")
    xall =fin1->maod(ncrec4,lat1:lat2,lon1:lon2); monthly
    xdust=fin1->maoddust(ncrec4,lat1:lat2,lon1:lon2); monthly
    xseas=fin1->maodseas(ncrec4,lat1:lat2,lon1:lon2); monthly
    xsulf=fin1->maodsulf(ncrec4,lat1:lat2,lon1:lon2); monthly
    xoc  =fin1->maodoc(ncrec4,lat1:lat2,lon1:lon2); monthly
    xbc  =fin1->maodbc(ncrec4,lat1:lat2,lon1:lon2); monthly

    yall =fin2->maod(ncrec4,lat1:lat2,lon1:lon2); monthly
    ydust=fin2->maoddust(ncrec4,lat1:lat2,lon1:lon2); monthly
    yseas=fin2->maodseas(ncrec4,lat1:lat2,lon1:lon2); monthly
    ysulf=fin2->maodsulf(ncrec4,lat1:lat2,lon1:lon2); monthly
    yoc  =fin2->maodoc(ncrec4,lat1:lat2,lon1:lon2); monthly
    ybc  =fin2->maodbc(ncrec4,lat1:lat2,lon1:lon2); monthly

    obs@_FillValue = -9999

    ave_x0 = wgt_areaave(obs, clat, 1.0, 1)
    ave_x1 = wgt_areaave(xall,clat, 1.0, 1)
    ave_x2 = wgt_areaave(yall,clat, 1.0, 1)
   ;print(modis_aod_ave)
    if (yr.lt.9999) then
      ave_x0=modis_aod_ave(yr-2003)
    else
      ave_x0=avg(modis_aod_ave)
    end if

    n0=-70 ; =>70S
    n1=20
   ;printVarSummary(xall)
   ;printVarSummary(clat)
    if (mon.eq.5) then
    do nn=0,6
     znl5(0,nn)=wgt_areaave(xall({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
     znl5(4,nn)=wgt_areaave(xdust({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
     znl5(1,nn)=wgt_areaave(xseas({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
     znl5(5,nn)=wgt_areaave(xsulf({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
     znl5(2,nn)=wgt_areaave(xoc({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
     znl5(3,nn)=wgt_areaave(xbc({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
    end do
    end if

    if (mon.eq.9) then
    do nn=0,6
     znl9(0,nn)=wgt_areaave(xall({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
     znl9(4,nn)=wgt_areaave(xdust({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
     znl9(1,nn)=wgt_areaave(xseas({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
     znl9(5,nn)=wgt_areaave(xsulf({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
     znl9(2,nn)=wgt_areaave(xoc({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
     znl9(3,nn)=wgt_areaave(xbc({n0+nn*n1:n0+(nn+1)*n1-1},:),clat({n0+nn*n1:n0+(nn+1)*n1-1}),1.0,0)
    end do
    end if
   ;print(znl5)
  end do ;yr
  end do ;mo

    fpng= "bar_znl_"+sprinti("%0.4i",yr2)+"_"+sprinti("%0.2i",mon)+".png"

  if (first.eq.1) then
   ;wks = gsn_open_wks("x11", "contour_map")   ;-- open a workstation
    wks = gsn_open_wks("pdf",get_script_prefix_name())
   ;wks = gsn_open_wks("png",fpng)
    plot_with_xgrid = new(2,graphic) ; create a plot panel
    lat_20=(/"60S","40S","20S","EQ","20N","40N","60N"/)
    xdim=dimsizes(lat_20)
;---Arrays to hold the bar values and the bars themselves.
    dum0 = new(xdim,graphic)
    dum1 = new(xdim,graphic)
    dum2 = new(xdim,graphic)
    dum3 = new(xdim,graphic)
    dum4 = new(xdim,graphic)
    dum5 = new(xdim,graphic)
    dum6 = new(xdim,graphic)
    dum7 = new(xdim,graphic)
    dum8 = new(xdim,graphic)
    dum9 = new(xdim,graphic)
    colors=(/"blue", "yellow", "black", "red","cyan"/)
    names=(/"Sea Salt","OC","BC","Dust","Sulfate"/)

    first=first+1
  end if ; first=1

  res                          = True
  res@gsnMaximize              = True    ; Maximize plot in frame
  res@gsnDraw                  = False   ; Don't draw
  res@gsnFrame                 = False   ; Don't advance frame
  res@vpWidthF                 = 0.4     ; Make long and
  res@vpHeightF                = 0.4     ; narrow
;---Set axes limits. Add extra space for X max.
  res@trYMinF                  = 0.0
 ;res@trYMaxF                  = (max(d2) + max(d1)+max(d3) + max(d4))*1.2  
  res@trYMaxF                  = (max(znl5))*1.2  
  res@trYMaxF                  = .42
  res@trXMinF                  = 0
  res@trXMaxF                  = xdim + 1
;---Put city labels on X axis
  res@tmXBMode                 = "Explicit"
  res@tmXBValues               = ispan(1,xdim,1)
  res@tmXBLabels               = lat_20
 ;res@tmXBLabelJust            = "BottomCenter"
  res@tmXBLabelFontHeightF     = 0.01   ; make labels smaller
  res@gsnScale                 = True    ; Make labels on axes same size

;--Turn off X axis tickmarks
  res@tmXBMajorLengthF         = 0.
  res@tmXBMajorOutwardLengthF  = 0.  
  res@tmYLMajorLengthF         = 0.01
  res@tmYLMajorOutwardLengthF  = 0.01

  res@tmXTOn                   = False   ; Turn off top tickmarks
  res@tiMainString  = "Zonal Mean AOD Components May vs. Sep"


;---Create blank plot without Y grid lines
  plot_without_xgrid = gsn_csm_blank_plot(wks,res)

;---Create blank plot with Y grid lines
  res@tmYMajorGrid                = True    ; Turn on grid lines
  res@tmYMajorGridLineDashPattern = 2       ; Dashed lines
  res@tmYMajorGridThicknessF      = 1.0     ; default is 2 (?)
  plot_with_xgrid(mmm) = gsn_csm_blank_plot(wks,res)

;---Set some resources for the bars.
  bres           = True
  bres@gsEdgesOn = True      ; Outline the polygons (bars)
;
; Loop through each city and attach bar to plot that
; doesn't have grid lines.

  c1=new(1,float)
  c2=new(1,float)
  c3=new(1,float)
  c4=new(1,float)
  dx1=0.30
  dx2=0.02
  do i=1,xdim
    bres@gsFillColor          = colors(0)
    c1=0
    c2=znl5(1,i-1)
    xbar5     = (/i-dx1,i-dx1,i-dx2,i-dx2,i-dx1/)
    ybar5     = (/c1,c2,c2,c1,c1/)
    dum0(i-1) = gsn_add_polygon(wks,plot_without_xgrid,xbar5,ybar5,bres)

    c3=0
    c4=znl9(1,i-1)
    xbar9     = (/i+dx2,i+dx2,i+dx1,i+dx1,i+dx2/)
    ybar9     = (/c3,c4,c4,c3,c3/)
    dum1(i-1) = gsn_add_polygon(wks,plot_without_xgrid,xbar9,ybar9,bres)

    bres@gsFillColor          = colors(1)
    c1=c2
    c2=znl5(2,i-1)+znl5(1,i-1)
    ybar5     = (/c1,c2,c2,c1,c1/)
    dum2(i-1) = gsn_add_polygon(wks,plot_without_xgrid,xbar5,ybar5,bres)

    c3=c4
    c4=znl9(2,i-1)+znl9(1,i-1)
    ybar9     = (/c3,c4,c4,c3,c3/)
    dum3(i-1) = gsn_add_polygon(wks,plot_without_xgrid,xbar9,ybar9,bres)
    
    bres@gsFillColor          = colors(2)
    c1=c2
    c2=znl5(3,i-1)+znl5(2,i-1)+znl5(1,i-1)
    ybar5     = (/c1,c2,c2,c1,c1/)
    dum4(i-1) = gsn_add_polygon(wks,plot_without_xgrid,xbar5,ybar5,bres)

    c3=c4
    c4=znl9(3,i-1)+znl9(2,i-1)+znl9(1,i-1)
    ybar9     = (/c3,c4,c4,c3,c3/)
    dum5(i-1) = gsn_add_polygon(wks,plot_without_xgrid,xbar9,ybar9,bres)
    
    bres@gsFillColor          = colors(3)
    c1=c2
    c2=znl5(4,i-1)+znl5(3,i-1)+znl5(2,i-1)+znl5(1,i-1)
    ybar5     = (/c1,c2,c2,c1,c1/)
    dum6(i-1) = gsn_add_polygon(wks,plot_without_xgrid,xbar5,ybar5,bres)

    c3=c4
    c4=znl9(4,i-1)+znl9(3,i-1)+znl9(2,i-1)+znl9(1,i-1)
    ybar9     = (/c3,c4,c4,c3,c3/)
    dum7(i-1) = gsn_add_polygon(wks,plot_without_xgrid,xbar9,ybar9,bres)

    bres@gsFillColor          = colors(4)
    c1=c2
    c2=znl5(5,i-1)+znl5(4,i-1)+znl5(3,i-1)+znl5(2,i-1)+znl5(1,i-1)
    ybar5     = (/c1,c2,c2,c1,c1/)
    dum8(i-1) = gsn_add_polygon(wks,plot_without_xgrid,xbar5,ybar5,bres)

    c3=c4
    c4=znl9(5,i-1)+znl9(4,i-1)+znl9(3,i-1)+znl9(2,i-1)+znl9(1,i-1)
    ybar9     = (/c3,c4,c4,c3,c3/)
    dum9(i-1) = gsn_add_polygon(wks,plot_without_xgrid,xbar9,ybar9,bres)

end do
;**********************************************************
; add labelbar to plot
;********************************************************** 
nboxes =dimsizes(colors)
  getvalues plot_with_xgrid(mmm)                ; get plot size for use in creating labelbar
  "vpXF"      : vpx
  "vpYF"      : vpy
  "vpHeightF" : vph
  "vpWidthF"  : vpw
  end getvalues

;  lbw    = 0.15 * vpw           ; Make labelbar size a fraction of the plot.
;  lbh    = 0.2 * vph
   
  lbres                    = True          ; labelbar only resources
  lbres@lbPerimOn          = False
  lbres@vpWidthF           = 0.15 * vpw     ; labelbar width
  lbres@vpHeightF          = 0.15 * vph     ; labelbar height
  lbres@lbBoxMajorExtentF  = 0.8          ; puts space between color boxes
  lbres@lbFillColors       = colors ; labelbar colors
  lbres@lbMonoFillPattern  = True          ; Solid fill pattern
  lbres@lbLabelFontHeightF = 0.013          ; font height. default is small
  lbres@lbJustification    = "TopLeft"
;  lbres@lbLabelJust        = "CenterLeft"  ; left justify labels
   
  gsn_labelbar_ndc(wks,nboxes,names,0.10,0.66,lbres)

  overlay(plot_with_xgrid(mmm),plot_without_xgrid)
  mmm=mmm+1
 ;draw(plot_without_xgrid)
 ;frame(wks)


;************************************************
; create panel
;************************************************
  resP                  = True                   ; modify the panel plot
  resP@gsnPanelMainFontHeightF = 0.012
  resP@gsnPanelLabelBar = False                   ; add common colorbar
  gsn_panel(wks,plot_with_xgrid,(/1,2/),resP)               ; now draw as one plot
end
