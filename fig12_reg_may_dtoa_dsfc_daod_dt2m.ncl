load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin
; plot AOD & RF in ProgAer and diff from ClimAer in May 2004
; (1) full array is read in, (2) calculate is done globlly, (3) plot is done regionally

  mo1=5
  mo2=9
  modis_aod_ave=new(17,float)
  pth_clim ="/scratch2/BMC/gsd-hpcs/Shan.Sun/clim_togo_exp/COMROOT/clim0/"
  pth_chm = "/scratch2/BMC/gsd-fv3-test/Shan.Sun/wgne_chem_exp/COMROOT/chm0/"
; pth_ctrl0="/scratch1/BMC/wrfruc/Shan.Sun/wfw_wgne_ctrl_exp/COMROOT/noa00/"
  ncrec0=0 ; monthly
  ncrec4=4 ; monthly

  ymin=0
  ymax=1
  first=1
  rad    = 4.0*atan(1.0)/180.0
  nn=0
 ;do mon=mo1,mo2,mo2-mo1
  do mon=5,5
    if (mon.eq.5) then
      amon="May "
      lat1=0;
      lat2=179;
      lon1=0;
      lon2=359;
      mplat1=91;
      mplat2=125;
      mplon1=-29;
      mplon2=30;

    ; mplat1=lat1;
    ; mplat2=lat2;
    ; mplon1=lon1;
    ; mplon2=lon2;
      yr=2004
    end if
    if (mon.eq.9) then
      amon="Sep "
      lat1=0;
      lat2=179;
      lon1=0;
      lon2=359;
      mplat1=61;
      mplat2=90;
      mplon1=1;
      mplon2=30;
      yr=2010
    end if
    patha=pth_clim
    pathb=pth_chm
    sfcdata1= "gfs."+sprinti("%0.4i",yr)+sprinti("%0.2i",mon)+"01/00/var_wk_1x1.nc"
    fin1= addfile(patha+sfcdata1,"r");
    fin2= addfile(pathb+sfcdata1,"r");
    lonn=fin1->lon(lon1:lon2);
    latn=fin1->lat(lat1:lat2);
    clat   = cos(latn*rad)
    clat!0="lat"  ; name the dim
    clat&lat=latn ; assign the coord var

    do nvar=0,3; AOD, RF, sfc, T2m
      if (nvar.eq.0) then
        var=" AOD "
        x91=fin1->maod(ncrec4,lat1:lat2,lon1:lon2); monthly
        x92=fin2->maod(ncrec4,lat1:lat2,lon1:lon2); monthly

        x91!0="lat"  ; name the dim
        x91&lat=latn ; assign the coord var
        x91!1="lon"  ; name the dim
        x91&lon=lonn ; assign the coord var
        x92!0="lat"  ; name the dim
        x92&lat=latn ; assign the coord var
        x92!1="lon"  ; name the dim
        x92&lon=lonn ; assign the coord var
      end if
      if (nvar.eq.1) then
        var=" RF TOA "
        x11=fin1->dswrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
        x21=fin1->uswrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
        x31=fin1->ulwrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
    
        x12=fin2->dswrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
        x22=fin2->uswrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
        x32=fin2->ulwrf_avetoa(ncrec4,lat1:lat2,lon1:lon2); monthly
    
        x91=x11-x21-x31
        x92=x12-x22-x32
      end if ;nvar
      if (nvar.eq.2) then
        var=" Rad SFC "
        x41=fin1->dswrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
        x51=fin1->uswrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
        x61=fin1->dlwrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
        x71=fin1->ulwrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    
        x42=fin2->dswrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
        x52=fin2->uswrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
        x62=fin2->dlwrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
        x72=fin2->ulwrf_ave(ncrec4,lat1:lat2,lon1:lon2); monthly
    
        x91=x41+x61-x51-x71
        x92=x42+x62-x52-x72
      end if ;nvar
      if (nvar.eq.3) then
        var=" T2m "
        x91=fin1->tmp2m(ncrec4,lat1:lat2,lon1:lon2); monthly
        x92=fin2->tmp2m(ncrec4,lat1:lat2,lon1:lon2); monthly
        x91=x91-273.15
        x92=x92-273.15
      end if ;nvar

      ave_x11= wgt_areaave(x91({mplat1-90:mplat2-90},{0:29}),   clat({mplat1-90:mplat2-90}), 1.0, 1)
      ave_x12= wgt_areaave(x91({mplat1-90:mplat2-90},{330:359}),clat({mplat1-90:mplat2-90}), 1.0, 1)
      ave_x21= wgt_areaave(x92({mplat1-90:mplat2-90},{0:29}),   clat({mplat1-90:mplat2-90}), 1.0, 1)
      ave_x22= wgt_areaave(x92({mplat1-90:mplat2-90},{330:359}),clat({mplat1-90:mplat2-90}), 1.0, 1)

      ave_x1=(ave_x11+ave_x12)*.5
      ave_x2=(ave_x21+ave_x22)*.5
      print(ave_x1)
      print(ave_x2)

  if (first.eq.1) then
   ;wks = gsn_open_wks("x11", "contour_map")   ;-- open a workstation
    wks = gsn_open_wks("png",get_script_prefix_name())  ; for latex
   ;wks = gsn_open_wks("pdf",get_script_prefix_name())
    plot = new(8,graphic) ; create a plot panel

    res                     = True
    res@gsnDraw             = False            ; don't draw
    res@gsnFrame            = False            ; don't advance frame
    res@cnInfoLabelOn       = False            ; turn off cn info label
    res@cnFillOn            = True             ; turn on color
    res@mpFillOn            = False             ; turn on color
    res@cnFillPalette     = "nrl_sirkes";"temp_diff_18lev"; "precip3_16lev"; "sunshine_9lev"; "temp_diff_18lev"; "ncl_default"   ; define color map for contours
    res@lbLabelBarOn      = True
    res@lbLabelFontHeightF  = 0.020   ; make labels smaller
    res@cnLinesOn         = False    ; turn off contour lines
    res@cnLineLabelsOn    = False    ; turn off line labels
   ;res@tiMainString      = ""
    res@gsnLeftString = ""
    first=first+1
  end if ;first=1

    res@mpMinLatF    = mplat1-90; always needed; Default: -90.0
    res@mpMaxLatF    = mplat2-90; always needed; 
    res@mpMinLonF    = mplon1; Default: -180.0
    res@mpMaxLonF    = mplon2

    ; delete(res@cnLevels)
    ; delete(res@cnFillColors)
    if (nvar.eq.0) then ; AOD
     ;res@cnLevels    = (/0.,2.,4.,6.,8.,10.,12.,14.,16./)   ; 9 num for TOA
     ;res@cnLevels    = (/.1,.2,.3,.4,.5,.6,.7,.8,.9,1./)   ; AOD
     ;res@cnLevels    = (/.1,.3,.5,.7,.9,1.1,1.3,1.5,1.7,1.9/)   ; AOD
      res@cnLevels    = (/.2,.4,.6,.8,1.0,1.2,1.4,1.6/)   ; 8 AOD
      num1="(a)"
      num2="(e)"
    end if
    if (nvar.eq.1) then ; TOA
      res@cnLevels = (/125.,150.,175.,200.,225.,250.,275.,300./) 
      res@cnLevels = (/-20.,0.,20.,40.,60.,80.,100.,120./) 
      num1="(b)"
      num2="(f)"
    end if;
    if (nvar.eq.2) then ; SFC
      res@cnLevels = (/125.,150.,175.,200.,225.,250.,275.,300./) 
      res@cnLevels = (/40.,65.,90.,115.,140.,165.,190.,215./) 
      num1="(c)"
      num2="(g)"
    end if;
    if (nvar.eq.3) then ; t2m
      res@cnLevels = (/.2,.4,.6,.8,1.0,1.2,1.4,1.6/)
      res@cnLevels = (/.6,1.2,1.8,2.4,3.0,3.6,4.2,4.8/)
      res@cnLevels = (/13,16,19,21,24,27,30,33/)
      res@cnLevels = (/14,17,20,23,26,29,32,35/)
      num1="(d)"
      num2="(h)"
    end if;
   ;res@cnFillColors= (/-1,1,2,4,6,8,9,10,12,14,16/); 11
    res@cnFillColors= (/-1,2,4,6,8,10,12,14,16/); 9
    res@gsnCenterStringFontHeightF = 0.024
    res@gsnRightStringFontHeightF = 0.022
  ; res@cnLevelSelectionMode = "ManualLevels"
    res@cnLevelSelectionMode = "ExplicitLevels"
    res@gsnCenterString = amon+sprinti("%0.4i",yr)+var+"ClimAer"  ; +var+obs
    res@gsnCenterString = num1+var+"ClimAer"  ; +var+obs
    res@cnFillPalette = "precip3_16lev"     ; set color map

    res@gsnRightString = sprintf("%4.2f",ave_x1)
    plot(nvar) = gsn_csm_contour_map(wks,x91,res)

   ;delete(res@cnLevels)
   ;delete(res@cnFillColors)
    if (nvar.eq.0) then ; AOD dif
     ;res@cnLevels = (/0.,2.,4.,6.,8.,10.,12.,14.,16./)   ; 9 num for TOA
     ;res@cnLevels = (/-80.,-60.,-40.,-20.,0.,20.,40.,60.,80./)   ; 9 num for TOA
     ;res@cnLevels = (/.1,.2,.3,.4,.5,.6,.7,.8,.9/)   ; AOD
     ;res@cnLevels =(/-.5,-.4,-.3,-.2,-.1,.1,.2,.3,.4,.5/) ; even levels
      res@cnLevels =(/-.4,-.3,-.2,-.1,.1,.2,.3,.4/) ; even levels
    end if
    if (nvar.eq.1) then ; TOA dif
     ;res@cnLevels = (/-80.,-60.,-40.,-20.,0.,20.,40.,60.,80./)   ; 9 num for TOA
     ;res@cnLevels =(/-25,-20,-15,-10,-5,5,10,15,20,25/) ; even levels
     ;res@cnLevels =(/-50,-40,-30,-20,-10,10,20,30,40,50/) ; even levels
     ;res@cnLevels =(/-5,-4,-3,-2,-1,1,2,3,4,5/) ; even levels
     ;res@cnLevels =(/-9,-7,-5,-3,-1,1,3,5,7,9/) ; even levels
     ;res@cnLevels =(/-12,-9,-6,-3,3,6,9,12/) ; even levels
      res@cnLevels =(/-8,-6,-4,-2,2,4,6,8/) ; even levels
    end if;
    if (nvar.eq.2) then ; SFC
      res@cnLevels =(/-8,-6,-4,-2,2,4,6,8/) ; even levels
    end if;
    if (nvar.eq.3) then ; t2m dif
      res@cnLevels =(/-.4,-.3,-.2,-.1,.1,.2,.3,.4/) ; even levels
      res@cnLevels =(/-1.2,-.8,-.4,-.2,.2,.4,.8,1.2/) ; even levels
    end if;
     ;res@cnFillColors = (/-1,1,2,4,6,8,9,10,12,14,16/); 11
      res@cnFillPalette= "BlAqGrWh2YeOrReVi22";
      res@cnFillPalette   = "BlRe"           ; set color map
      res@cnFillPalette= "BlWhRe"           ; 51 is white
     ;res@cnFillColors = (/18,31,35,42,46,-1,48,53,60,75,88/) ; even+1, 51 is white
      res@cnFillColors = (/31,35,42,46,-1,53,56,60,75/) ; even+1, 51 is white
    ; res@cnMinLevelValF  = -50.               ; min level
    ; res@cnMaxLevelValF  = 50.                ; max level
    ; res@cnLevelSpacingF = 10.                ; contour level spacing
    ; res@cnLevels       =(/-25,-20,-15,-10,-5,5,10,15,20,25/) ; even levels
    ; res@gsnMaximize = True                        ; maximize plot in frame

    res@gsnCenterString = amon+sprinti("%0.4i",yr)+var+"ProgAer minus ClimAer"
    res@gsnCenterString = num2+var+"ProgAer minus ClimAer"
    res@gsnRightString = sprintf("%4.2f",ave_x2-ave_x1)
    x91=x92-x91
    plot(nvar+4) = gsn_csm_contour_map(wks,x91,res)

  end do ;nvar
  end do ;mon

;************************************************
; create panel
;************************************************
  resP                  = True                   ; modify the panel plot
  resP@gsnPanelMainFontHeightF = 0.015
  resP@gsnPanelMainString = amon+sprinti("%0.4i",yr)
  resP@gsnPanelLabelBar = True                   ; add common colorbar
  resP@gsnPanelLabelBar = False                   ; add common colorbar
  resP@lbLabelFontHeightF  = 0.01                ; make labels smaller
  gsn_panel(wks,plot,(/2,4/),resP)               ; now draw as one plot
end
