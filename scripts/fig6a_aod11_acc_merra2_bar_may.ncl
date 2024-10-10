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
  file00=addfile("/home/Shan.Sun/helpme/land180.nc","r")
 mm=9
 mm=5
 if (mm.eq.5) then
  file0=addfile("/scratch1/BMC/gsd-fv3-dev/data_others/AOD/AOD_MODIS_2003_2019_1deg_daily/MODIS_AOD_May2003_2019.nc","r")
  file0b=addfile("/scratch1/BMC/gsd-fv3-dev/data_others/AOD/AOD_MERRA2_2003_2019_1deg_daily/merra2_aod_may2003_2019.nc","r")
  file1=addfile("/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm0/aer_daily_may_2003_2019.nc","r")
  file2=addfile("/scratch1/BMC/wrfruc/Shan.Sun/wgne_clim/clim0/aer_daily_may_2003_2019.nc","r")
 end if
 if (mm.eq.9) then
  file0=addfile("/scratch1/BMC/gsd-fv3-dev/data_others/AOD/AOD_MODIS_2003_2019_1deg_daily/MODIS_AOD_Sep2003_2019.nc","r")
  file0b=addfile("/scratch1/BMC/gsd-fv3-dev/data_others/AOD/AOD_MERRA2_2003_2019_1deg_daily/merra2_aod_sep2003_2019.nc","r")
  file1=addfile("/scratch1/BMC/wrfruc/Shan.Sun/wgne_chm/chm0/aer_daily_sep_2003_2019.nc","r")
  file2=addfile("/scratch1/BMC/wrfruc/Shan.Sun/wgne_clim/clim0/aer_daily_sep_2003_2019.nc","r")
 end if
  land=file00->veg(:,:)
  land=lonFlip(land) 
  time=file0->time
  dimt=dimsizes(time)
  lat=file0->lat
  time2=new(dimt, integer)
  time2=ispan(1,dimt,1)
  time2!0         = "time"
  time2@long_name = "time2"
  time2&time      = time2

  true=1 ; modis
  true=2 ; merra2
  read0=file1->aod550(:,:,:); filler

  tmp0=file0->MYD08_D3_6_1_AOD_550_Dark_Target_Deep_Blue_Combined_Mean(:,:,:)
  read0=doubletofloat(tmp0)
  read0b=file0b->TOTEXTTAU(:,:,:)
  read1=file1->aod550(:,:,:)
  read2=file2->aod550(:,:,:)
 ;printVarSummary(read0)
 
  read0=lonFlip(read0)
  read0b=lonFlip(read0b)
  read1=lonFlip(read1)
  read2=lonFlip(read2)

  land0=file1->aod550(:,:,:); filler
  land0b=file1->aod550(:,:,:); filler
  land1=file1->aod550(:,:,:); filler
  land2=file1->aod550(:,:,:); filler
  land0=lonFlip(land0)
  land0b=lonFlip(land0b)
  land1=lonFlip(land1)
  land2=lonFlip(land2)
  land0@_FillValue = -9999
  land0b@_FillValue = -9999
  land1@_FillValue = -9999
  land2@_FillValue = -9999

  ocn0=file1->aod550(:,:,:); filler
  ocn0b=file1->aod550(:,:,:); filler
  ocn1=file1->aod550(:,:,:); filler
  ocn2=file1->aod550(:,:,:); filler
  ocn0=lonFlip(ocn0)
  ocn0b=lonFlip(ocn0b)
  ocn1=lonFlip(ocn1)
  ocn2=lonFlip(ocn2)
  ocn0@_FillValue = -9999
  ocn0b@_FillValue = -9999
  ocn1@_FillValue = -9999
  ocn2@_FillValue = -9999

; print(dimt)
; printVarSummary(land)
; printMinMax(land, True)
; printVarSummary(read1)
; printVarSummary(land1)
; printVarSummary(ocn1)

  do n=0,dimt-1
  land0(n,:,:) = where(land.gt.0.,read0(n,:,:),-9999)
  land0b(n,:,:) = where(land.gt.0.,read0b(n,:,:),-9999)
  land1(n,:,:) = where(land.gt.0.,read1(n,:,:),-9999)
  land2(n,:,:) = where(land.gt.0.,read2(n,:,:),-9999)
   ocn0(n,:,:) = where(land.eq.0.,read0(n,:,:),-9999)
   ocn0b(n,:,:) = where(land.eq.0.,read0b(n,:,:),-9999)
   ocn1(n,:,:) = where(land.eq.0.,read1(n,:,:),-9999)
   ocn2(n,:,:) = where(land.eq.0.,read2(n,:,:),-9999)
  end do

  land0@_FillValue = -9999
  land0b@_FillValue = -9999
  land1@_FillValue = -9999
  land2@_FillValue = -9999
   ocn0@_FillValue = -9999
   ocn0b@_FillValue = -9999
   ocn1@_FillValue = -9999
   ocn2@_FillValue = -9999

  nnn=11 ; 12 areas
  rmseaod=new((/nnn,2/),float)
   accaod=new((/nnn,2/),float)
    aod17=new((/4,dimt/),float)
    prob=new((/nnn,2/),float)

  lat1=new(nnn,float)
  lat2=new(nnn,float)
  lon1=new(nnn,float)
  lon2=new(nnn,float)

;;      can west east SAm NAf SAf ME,Ind,As, NAt, SAt ,IO
;;  lat1=(/50.,25, 25, -35,  0, -30,10, 8,20, 0, -35, 0 /)
;;  lat2=(/70.,50, 50,   0, 30,  0, 30, 35,50, 35, 0,  25/)
;;  lon1=(/-160,-125,-95,-80,-20,10,30,70,100,-80,-35, 40/)
;;  lon2=(/-60., -95,-65,-35, 30,40,60,95,140,-10,20, 100/)

;;      can west east SAm NAf SAf ME,Ind,As, NAt, SAt ,IO
; lat1=(/50.,25, 25, -35,  0, -30,10, 8,20, 0, -35, 0 /)
; lat2=(/70.,50, 50,   0, 30,  0, 30, 35,50, 35, 0,  25/)
; lon1=(/-160,-125,-95,-80,-20,10,30,70,100,-80,-35, 40/)
; lon2=(/-60., -95,-65,-35, 30,50,60,95,140,-10,20, 100/)
; runid=(/"Canada","W US","E US","S Ameri","N Africa","S Africa","Mid East","Indian","E Asia","N At Ocean","S At Ocean","Indian Ocean"/)

  lat1=(/50.,25, 25, -35,  0, -30, 8,20, 0, -35,  0 /)
  lat2=(/70.,50, 50,   0, 30,  0,  35,50, 35, 0,  25/)
  lon1=(/-160,-125,-95,-80,-20,10,70,100,-80,-35, 40/)
  lon2=(/-60., -95,-65,-35, 40,40,95,140,-10,20, 100/)
  runid=(/"Canada","W US","E US","S Ameri","N Africa","S Africa","Indian","E Asia","N Atl Ocean","S Atl Ocean","Indian Ocean"/)

  aod17!1 = "time"
  aod17&time = time2
 ;printVarSummary(aod17)
  do n=0,nnn-1
    print(n)
    lat_sub=lat({lat1(n):lat2(n)})
    clat_sub=cos(lat_sub*rad)

    if (n .lt. 9) then
      x0=land0(:,{lat1(n):lat2(n)},{lon1(n):lon2(n)})
      x0b=land0b(:,{lat1(n):lat2(n)},{lon1(n):lon2(n)})
      x1=land1(:,{lat1(n):lat2(n)},{lon1(n):lon2(n)})
      x2=land2(:,{lat1(n):lat2(n)},{lon1(n):lon2(n)})
    else
      x0=ocn0(:,{lat1(n):lat2(n)},{lon1(n):lon2(n)})
      x0b=ocn0b(:,{lat1(n):lat2(n)},{lon1(n):lon2(n)})
      x1=ocn1(:,{lat1(n):lat2(n)},{lon1(n):lon2(n)})
      x2=ocn2(:,{lat1(n):lat2(n)},{lon1(n):lon2(n)})
    end if
    ;print(l1(0,:,:))
    ;printVarSummary(clat_sub)
    y0= wgt_areaave_Wrap(x0, clat_sub, 1.0, 1)
    y0b=wgt_areaave_Wrap(x0b,clat_sub, 1.0, 1)
    y1= wgt_areaave_Wrap(x1, clat_sub, 1.0, 1)
    y2= wgt_areaave_Wrap(x2, clat_sub, 1.0, 1)
   ;printVarSummary(y2)
    if (true.eq.1) then
     ;rmseaod(n,0)=dim_rmsd_Wrap(y1,y0)
     ;rmseaod(n,1)=dim_rmsd_Wrap(y2,y0)
      accaod(n,0)=escorc(y1,y0)
      accaod(n,1)=escorc(y2,y0)
    end if
    if (true.eq.2) then
     ;rmseaod(n,0)=dim_rmsd_Wrap(y1,y0b)
     ;rmseaod(n,1)=dim_rmsd_Wrap(y2,y0b)
      accaod(n,0)=escorc(y1,y0b);
      accaod(n,1)=escorc(y2,y0b);
    end if
    
    ;printVarSummary(y0)
    aod17(0,:)=y0
    aod17(1,:)=y0b
    aod17(2,:)=y1
    aod17(3,:)=y2
  ; print("aod0="+aod0(0:5)+" aod1="+aod1(0:5)+" aod2="+aod2(0:5))
    delete([/x0,x0b,x1,x2,lat_sub,clat_sub/])
  ; print(accaod)
  ; print(aod17)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  siglvl  = 0.05
  aveX    = avg (y1)
  aveY    = avg (y0b)
  varX    = variance (y1)
  varY    = variance (y0b)
  sX      = dimsizes (y1)
  sY      = dimsizes (y0b)
                                                   ; Following not used; FYI only
  diffXY  = aveX - aveY                            ; 4.378

  iflag   = True                                   ; population variance similar
  tval_opt= False                                  ; p-value only
  prob0 = ttest(aveX,varX,sX, aveY,varY,sY, iflag, True) 
  prob(n,0)=prob0(0,0)
  print(prob0)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  siglvl  = 0.05
  aveX    = avg (y2)
  aveY    = avg (y0b)
  varX    = variance (y2)
  varY    = variance (y0b)
  sX      = dimsizes (y2)
  sY      = dimsizes (y0b)
                                                   ; Following not used; FYI only
  diffXY  = aveX - aveY                            ; 4.378

  iflag   = True                                   ; population variance similar
  tval_opt= False                                  ; p-value only
  prob0 = ttest(aveX,varX,sX, aveY,varY,sY, iflag, True) 
  prob(n,1)=prob0(0,0)
  print(prob0)

  end do ; nnn
  print(prob)
  ; print(rmseaod)
  ; print(y0b)
  ; print(y1)
  ; print(y2)
    print(accaod)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; opt = 0  ; 1 indicates an unpaired t-test assuming equal variances
; prob = student_t(y2, y0b)
; print("p-value = " + prob)

   ;wks = gsn_open_wks("x11", "contour_map")   ;-- open a workstation
   ;wks = gsn_open_wks("eps",get_script_prefix_name())
   ;wks = gsn_open_wks("png",get_script_prefix_name())
    wks = gsn_open_wks("pdf",get_script_prefix_name())
  ; plot = new(nnn,graphic) ; create a plot panel

     sres = True
     sres@vpWidthF = 0.7
     sres@vpHeightF = 0.22
     sres@trXMinF = 1-0.6
     sres@trXMaxF = nnn+.6
     sres@vpXF = .15
     sres@gsnDraw = True
     sres@gsnFrame = False
     sres@gsnXYBarChart = True
     sres@gsnYRefLine   = 0 
     sres@gsnXYBarChartBarWidth = 0.30           ; change bar widths
     sres@tmXBMode          = "Explicit"         ; explicit labels
     sres@tmXBValues        = (/1,2,3,4,5,6,7,8,9,10,11/)
    ;sres@tmXBLabels = (/"Can","W US","E US","A4","A5","Jun","Jul","Aug","Sep","Oct","Nov","Dec"/)
    ;sres@tmXBLabels = (/"Can","W US","E US","S Am","N Afr","S Afr","Mid E","Indian","Asia","NA O","SA O","IO"/)
     sres@tmXBLabels=(/"Can","W US","E US","S Am","N Af","S Af","Ind","Asia","NAO","SAO","InO"/)
     sres@tmXBLabelFontHeightF = 0.0150
     sres@tmXTLabelFontHeightF = 0.0150
     sres@tmYLLabelFontHeightF = 0.0150
     sres@tiMainFont = "helvetica"
     sres@tmXBLabelFontHeightF = 0.0150
     sres@gsnCenterStringFontHeightF = 0.014
     sres@tiMainFontHeightF = 0.015
     ymin=-.4 
     ymax= 0.9
   if (mm.eq.5) then
     sres@gsnCenterString = "May"
     if (true.eq.1) then
       sres@tiMainString = "Daily AOD Correlation with MODIS in May 2003-2019"
     end if
     if (true.eq.2) then
       sres@tiMainString = "Daily AOD Correlation with MERRA2 in 2003-2019"
     end if
   end if
   if (mm.eq.9) then
     sres@gsnCenterString = "Sept"
  ;  if (true.eq.1) then
  ;    sres@tiMainString = "Daily AOD Correlation with MODIS in Sept 2003-2019"
  ;  end if
  ;  if (true.eq.2) then
  ;    sres@tiMainString = "Daily AOD Correlation with MERRA2 in Sept 2003-2019"
  ;  end if
   end if
     sres@trYMinF = ymin
     sres@trYMaxF = ymax
    ;sres@tiYAxisString = "(~S~o~N~C)"
     sres@tiYAxisString = ""
        
     sres@gsnXYBarChartColors = (/"red"/)	
    ;plot1 = gsn_csm_xy(wks,fspan(.775,11.775,nnn),rmseaod(:,0),sres)		; draw each time series
     plot1 = gsn_csm_xy(wks,fspan(.775,10.775,nnn),accaod(:,0),sres)		; draw each time series
     sres@gsnXYBarChartColors = (/"blue"/)					; drawn on the X-axis
    ;plot2 = gsn_csm_xy(wks,fspan(1.225,12.225,nnn),rmseaod(:,1),sres)
     plot2 = gsn_csm_xy(wks,fspan(1.225,11.225,nnn),accaod(:,1),sres)

     lbres                    = True          ; labelbar only resources
     lbres@vpWidthF           = 0.3           ; labelbar width
     lbres@vpHeightF          = 0.1           ; labelbar height
     lbres@lbBoxMajorExtentF  = 0.15          ; puts space between color boxes
     lbres@lbMonoFillPattern  = True          ; Solid fill pattern
     lbres@lbLabelFontHeightF = 0.015         ; font height. default is small
     lbres@lbLabelJust        = "CenterLeft"  ; left justify labels
     lbres@lbPerimOn          = False
     lbres@lgPerimColor 	 = "white"

   if (mm.eq.9) then
     colors = (/"red","blue"/)
     labels = (/"ProgAer","ClimAer"/)
     xpos = (/.25,.55/)
     do i=0,1
       lbres@lbFillColors = colors(i)
       gsn_labelbar_ndc(wks,1,labels(i),xpos(i),0.55,lbres)
     end do	
   end if
	
     frame(wks)      
end
